# Third-party code

Both libraries are by the CAMeL Lab and released under the MIT License (see each folder's `LICENSE`).
They are vendored here with small local changes.

| Folder | Upstream | Base commit | Local changes |
|---|---|---|---|
| `camel_parser/` | https://github.com/CAMeL-Lab/camel_parser | `be07811` | `src/data_preparation.py`: absolute `src.utils` import made relative so the package can be imported from this repo. Models, tests and sample data removed. |
| `conllx_df/` | https://github.com/CAMeL-Lab/conllx_df | `d204240` | Modules moved from `src/conllx_df/` to `src/`; `conll_utils.Token` gained a `lemma` field (filled from the `LEMMA` column). Tests and data removed. |
