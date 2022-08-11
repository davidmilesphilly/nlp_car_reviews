from bs4 import BeautifulSoup as bs
import requests

r = requests.get("https://github.com/davidmilesphilly/nlp_car_reviews/blob/74dee59c1754bf0fbfefb88237cc49974e2398f8/Scraped_Car_Review_dodge.csv")

soup = bs(r.content, features="html.parser")

contents = soup.prettify
# print(contents)
info_box = soup.find(class_ = "sc-bdxVC sc-kwMEKh cKNOET dUsjCJ")
print(info_box)
