# Covid Mask Sentiment

## Contributing to the project

* Make sure pre-commit hooks are installed by running `pre-commit install` in the root directory (one time set up).
* Obtain Twitter developer credentials [here](https://developer.twitter.com).
* Database URL = \<dialect\>+\<driver\>://\<user\>:\<password\>@host:port/database
* Create and fill secrets file using `cp .envrc.example .envrc`
* Give direnv permission to read `.envrc` by running `direnv allow`
* Install all dependencies using `pip install -r requirements.txt`

Start the Docker container by running `sh ./start_app.sh`.

## Annotation
A custom annotation tool was used to label the 2,500 tweets as ***positive***, ***negative*** or ***neutral***.
* **Positive:** wearing masks during the pandemic is helpful in diminishing the spread of covid-19.

* **Negative:** wearing masks is ineffective or harmful for the spread of covid-19

* **Neutral:** tweet mentions key words but does not express any sentiment.  
e.g. poses a question, product marketing, quoting any text or reference without adding an opinion, sentence unintelligible, etc.

*Note:* A small number of tweets were dropped from the analysis if the sentence was unintelligible or contained languages other than English that made sentiment impossible to assign.
