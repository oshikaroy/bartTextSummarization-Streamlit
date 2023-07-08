# bartTextSummarization-Streamlit

This project summarizes long text using the BART model (which is a Natural Language Processing model).

**Add Model path to lsaApproach.py**
```
terms_and_conditions_pipeline = pipeline(
                task='summarization',
                model="sshleifer/distilbart-xsum-12-3", //HERE
                tokenizer="sshleifer/distilbart-xsum-12-3" //HERE
            )
```
**To run:**

run lsaApproach.py on streamlit
