### MLESAC extension and ε sensitivity

Classic RANSAC picks the model with the **most inliers** under the threshold while MLESAC selects the model that **minimizes a truncated residual cost**: points closer than threshold contribute their distance, and points beyond contribute a constant penalty gamma. Standard RANSAC often changes its chosen plane noticeably when threshold increases, because a slightly larger threshold can suddenly turn many borderline points into inliers and shift the best hypothesis. MLESAC was typically **less sensitive to threshold**, since it still prefers planes that keep inliers **tight** instead of only maximizing the inlier count.

**Advantages of MLESAC:** more stable model selection under noisy data, usually less sensitive to threshold, and often produces a better-fitting plane when multiple hypotheses have similar inlier counts.
**Disadvantages:** still depends on hyperparameters, slightly higher computation per hypothesis (cost evaluation instead of only counting), and if gamma is poorly chosen the benefit can shrink.

---

### Preemptive RANSAC evaluation (M and B)

I implemented **Preemptive RANSAC** by sampling **M hypotheses** up front, evaluating them in batches of **B points**, and repeatedly pruning to keep only the best hypotheses after each batch. I evaluated different combinations of **M** and **B** on the exercise data for both scoring styles: (1) classic preemptive scoring via outlier counting (“RANSAC scoring”), and (2) **MSAC/MLESAC-style truncated cost** (“MLESAC scoring”). Across parameterizations, **M controls the time budget**: small M runs fast but can miss good hypotheses and produce unstable planes, while large M increases runtime but improves reliability because good models are more likely to be sampled. **B controls pruning aggressiveness**: small B prunes earlier and is faster, but can incorrectly discard good hypotheses if early batches are unrepresentative; larger B is usually more stable but costs more runtime.

For the “time budget” visualization, I compared the estimated plane for **three M values** (small/medium/large). The results show that increasing M generally makes the plane estimate and masks more consistent, while small M can lead to noisier or partially incorrect fits.

**Advantages of Preemptive RANSAC:** often faster than standard RANSAC at similar quality due to early rejection of bad hypotheses, and can be combined with MLESAC-style scoring for more stability.
**Disadvantages:** more hyperparameters (M, B) to tune, risk of pruning away the correct model when B is too small, and slightly more implementation complexity; using MLESAC scoring adds some extra computation and needs a reasonable gamma.
