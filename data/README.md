# RQ1 public dataset

The RQ1 experiment uses the **Wine Quality** dataset from the UCI Machine
Learning Repository.

- Official dataset page: `https://archive.ics.uci.edu/dataset/186/wine+quality`
- DOI: `10.24432/C56S3T`
- Authors: Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos,
  José Reis
- License shown by UCI: Creative Commons Attribution 4.0 International
- Downloaded: 2026-07-16
- Archive SHA-256:
  `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4`

The ZIP is kept unchanged in `raw/wine-quality.zip`; the experiment reads the
red and white CSV files directly from the archive.

## Augmentation construction

For each repeated train/validation/test split, the first two physicochemical
columns form the buyer's input table. Each of the remaining nine columns is
treated as a separate external table keyed by a synthetic immutable row ID.
Joining an external table therefore corresponds exactly to adding one feature
without changing rows. Candidate tables are ranked using training-only target
correlation, serving as a transparent small-scale replacement for Metam's data
profiles and likelihood scores.

This is a public-data, reduced-scale reproduction of the paper's RQ1 protocol;
it is not the unavailable 69K-dataset NYC market used in the original figure.
