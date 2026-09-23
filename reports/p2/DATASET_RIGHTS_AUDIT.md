# p2 dataset and rights audit

Checked 2026-09-23 from official sources.

## PixelRec50K

- Official project: https://github.com/westlake-repl/PixelRec at HEAD `a4e4a01b01aba96f746ae66de9eb35a88484a681`.
- Official dataset terms: https://github.com/westlake-repl/PixelRec/blob/main/dataset/LICENSE.
- The terms allow access exclusively for noncommercial research and education and state that no rights are granted to copy, modify, publish, distribute, or commercialize the dataset.
- Official README links interactions, item information, covers, and publisher-extracted text/image vectors. It describes PixelRec50K as 989,494 interactions, 50,000 users, and 82,865 items.
- Local interaction SHA-256: `638b53ec100f760cb9bd540c361f6d6e3617c81b1c054ced63fffa41da909e4d`.
- Local publisher image-vector source SHA-256: `b74fb95313afe53bfc432ef918f732b731ad64728887a2890417401e2ce19779`.
- Local aligned publisher image array SHA-256: `bb114e3d89e3983aa3798f72e0d3756f535c881366a8bacb65e44f70a9c301ae`.
- Raw covers, raw data, publisher features, local arrays, checkpoints, and recommendation lists remain outside Git and are not redistributed.
- Raw-cover extraction, transformation, and newly derived cover features are `HOLD`.

## MovieLens 32M

- Official page: https://grouplens.org/datasets/movielens/32m/.
- Official README: https://files.grouplens.org/datasets/movielens/ml-32m-README.html.
- Research use is allowed with acknowledgment; redistribution is allowed under the same conditions; commercial or revenue-bearing use requires permission.
- Required citation: Harper and Konstan, 2015, *The MovieLens Datasets: History and Context*.
- p2 use is behavioral/sequence transfer only. No visual multimodal replication claim is allowed.

## MicroLens and p1

MicroLens protected p1 outcomes are excluded from p2 selection and inference. The p1 post-hoc manifest and narrative are retained only as immutable hypothesis provenance. Synthetic or nonprotected rows may be used for engineering tests.

## Gate

PixelRec interaction and publisher image-feature use passes for local noncommercial p2 research under the stated restrictions. The p2 text-feature identity remains open. No data or feature bytes may be committed.
