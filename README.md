# Repository for [arXiv:2411.10227](https://arxiv.org/abs/2411.10227)

In this paper we use six corpora in 3 different languages accounting for different degrees of morphological isolation and agglutination. Their information is summarized in the table below.

| Corpus ID | Language | Source | Availability | Access |
|----------|----------|----------|----------|----------|
| SPGC    | English    | Books    | Yes    | [Publication](https://doi.org/10.3390/e22010126)    |
| SPA    | Spanish    | Online media    | Restricted    | [Mark Davies website](https://www.corpusdelespanol.org/web-dial/)    |
| TRCC100    | Turkish    | Web scrapping    | Yes    | [Publication](https://aclanthology.org/2020.acl-main.747/) and [direct download](https://metatext.io/datasets/cc100-turkish)    |
| TwEN    | English    | Twitter    | In preparation    | -    |
| TwES    | Spanish    | Twitter    | In preparation    | -    |
| TwTR    | Turkish    | Twitter    | In preparation    | -    |

To compile corpora and extract entropy and type-token ratio metrics we used the Python codes explained below.

| Code | Details |
|----------|----------|
| [Metrics computation](corpora_analysis_general.py)    | Computing type-token ratio and entropy (using both PI and NSB estimators) given an aggregated corpus, i.e., a compilation of all the documents in one   |
| [Corpora filtering](function_clean_text.py)    | Text cleaning (removing punctuation, digits, lowercasing)   |



