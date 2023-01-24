# FAKE NEWS DETECTOR

**Model for detecting whether a news article is fake or true based on the title and text body.**

It uses Logistic Regression and Vectorization to determine fakeness through the title and first 50 non-trivial words and categorizes them into either 0 (True) or 1 (Fake)

Trained on the [Fake News](https://www.kaggle.com/competitions/fake-news/data "Go to site") on kaggle

Accuracy on train set: ~93.02%

Accuracy on test set: ~92.88%

## To try it out:

1. Install requirements

```
  pip install -r requirements.txt
```

2. Update the TITLE and TEXT in main.ipynb for the news title and body
   eg:

   ```
   TITLE = "Keiser Report: Meme Wars (E995)"
   TEXT = "For the first time in history, we’re filming a panoramic video from the station. It means you’ll see everything we see here, with your own eyes. That’s to say, you’ll be able to feel like real cosmonauts' - Borisenko to RT. Video presented by RT in collaboration with the Russian space agency Roscosmos and the rocket and space corporation Energia More on our project website: space360.rt.com"
   ```
3. Run the code blocks in main.ipynb
