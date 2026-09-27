# Configuration

Home of the **Experiment Contract**: the single file that freezes every experimental choice before the
first fit and whose hash is recorded in the manifest of every run that uses it.

It must fix, at minimum: window bounds and warm-up buffer; feature cutoff, decision deadline and
execution times; the price fields; the label rule; fold layout and refit cadence; the feature list per
experiment arm; the news lag grid and the primary lag; in-fold preprocessing and imputation; model
family and regularisation grid; the selection metric and the primary comparison; the bootstrap
specification; the trial budget; the policy thresholds; and the cost scenarios.

A choice absent from the contract at fit time is not pre-registered, whatever a document claims
afterwards.

The design notes behind these requirements live outside version control; see the README section on
where the thinking lives.
