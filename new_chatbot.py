import sys
import os
import re
import string
import requests
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def search_google(query):
	while(1):
	  url='https://www.google.com/search?q='+query

	  page=requests.get(url)
	  #print('Obtained page',page)
	  soup=BeautifulSoup(page.content,'html.parser')

	  links=soup.findAll('a')
	  all_links=[]

	  for  link in links:
	    link_href=link.get('href')
	    if 'url?q=' in link_href and not 'webcache' in link_href:
	      all_links.append((link.get('href').split('?q=')[1].split('&sa=U')[0]))

	  flag=False
	  for link in all_links:
	    if 'https://en.wikipedia.org/wiki/' in link:
	      wiki=link
	      flag=True
	      break

	  div0=soup.find_all('div',class_='kvKEAb')
	  div1=soup.find_all('div',class_='Ap5OSd')
	  div2=soup.find_all('div',class_='nGphre')
	  div3=soup.find_all('div',class_='BNeawe iBp4i AP7Wnd')

	  if(len(div0)!=0):
	    ans=div0[0].text

	  elif(len(div1)!=0):
	    ans=div1[0].text+'\n'+div1[0].find_next_sibling('div').text

	  elif(len(div2)!=0):
	    ans=div2[0].find_next('span').text+'\n'+div2[0].find_next('div',class_='kCrYT').text

	  elif(len(div3)!=0):
	    ans=div3[1].text

	  elif(flag):
	    page2=requests.get(wiki)
	    soup=BeautifulSoup(page2.text,'html.parser')
	    title=soup.select('#firstHeading')[0].text

	    para=soup.select('p')

	    for par in para:
	      if bool(par.text.strip()):
	        ans=title+'\n'+par.text
	        break
	  else:
	    ans='Could not search given query'

	  return ans