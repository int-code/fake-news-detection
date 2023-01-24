import pandas as pd
from setup import stem, model, titlevectorizer, vectorizer

def check(news):
    title = stem(news['title'])
    text = stem(news['text'])
    print(title, text)
    title = titlevectorizer.transform([news['title']])
    title = pd.DataFrame.sparse.from_spmatrix(title)
    # print(title)

    # print("hsdh")
    text = vectorizer.transform([news['text']])
    text = pd.DataFrame.sparse.from_spmatrix(text)
    text.columns = [x for x in range(50,100)]
    title = pd.merge(title, text, left_index=True, right_index=True)
    print(title)
    res = model.predict(title)
    print(res)


check({
    'title':"Can Republicans Expunge Donald Trump’s Impeachments?",
    'text':'Republicans unhappy with the impeachments of former President Donald Trump have for years been vocal about their outrage. But during the last Congress, a group of more than two dozen lawmakers in the House went even further, introducing resolutions to “expunge” the impeachments. And now that Republicans have wrested back the majority in the chamber, House Speaker Kevin McCarthy says he’s willing to consider such an effort. “I understand why individuals want to do it, and we’d look at it,” McCarthy, California Republican, said in response to a question earlier this month during his first weekly news conference. But his comments raised questions of their own. Namely: Is expunging Trump’s impeachments even possible?The answer, constitutional experts say, is complicated.For one thing, it would be unprecedented in this context.According to the House historian, impeachment proceedings have been initiated more than 60 times in U.S. history against office holders across the federal government, and less than a third of the proceedings have led to full impeachments. That figure includes 15 federal judges, an 18th century senator and a 19th century Cabinet official along with, most notably, President Andrew Johnson in 1868, President Bill Clinton in 1998 and Trump. Eight impeachments – all of them of federal judges – have led to convictions and removals from office by the Senate. None has ever been expunged.'
})