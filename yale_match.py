# -*- coding: utf-8 -*-
"""yale_match.ipynb

Automatically generated using Jupyter 

Original file is located at
    https://colab.research.google.com/drive/1GHT70FjQwHZZ5xKrKHLfzMlcqGBb0ZCK
"""

import pandas as pd
import numpy as np
import csv
import heapq 
from random import randint


#compatibility of two inviiduals (rows)  
def compatibility(row1, row2): 

  #year differential 
   A = (4 - (abs(int(row1['year']) - int(row2['year'])))) * 1.5

   #locs in common  
   loc_list_1 = str(row1['date_loc']).split(';')
   loc_list_2 = str(row2['date_loc']).split(';')
   B = 0
   for loc in loc_list_1:
     if loc in loc_list_2:
       B += 0.25 

   #events in common  
   events_list_1 = str(row1['date_events']).split(';')
   events_list_2 = str(row2['date_events']).split(';')
   C = 0 
   for event in events_list_1:
     if event in events_list_2:
       C += 0.25 

   #myers-brigg compatiability 
   mb_list = ['ISFP','INFP','ESFP','ESTP','ISTP','INTP','ENFP','INFJ','INTJ','ENFJ','ISTJ','ENTP','ESTJ','ENTJ','ESFJ','ISFJ', 'other']
   personality1 = 'other'
   personality2 = 'other'
   for s in mb_list:
     if s in str(row1['personality']):
       personality1 = s
     if s in str(row2['personality']):
       personality2 = s 
   if personality1 == 'other' or personality2 == 'other':
     D = 0
   else:
     D = mb[personality1][personality2] * 0.8


   looking_list_1 = str(row1['looking_for']).split(';')
   looking_list_2 = str(row2['looking_for']).split(';')
   common = list(set(looking_list_1).intersection(looking_list_2))
   E = 0
   if len(common) == 0:
     E = -100 
   else: 
    #filter out those people who check all boxes 
    if len(looking_list_1) <= 1 or len(looking_list_2) <= 1:
       E = len(common) * 5
    else:
       E = len(common) * 2.5

   #other factors 
   text_partner_desc_1 = str(row1['TEXT_looking_partner']).lower()
   text_partner_desc_2 = str(row2['TEXT_looking_partner']).lower()
   text_desc_1 = str(row1['TEXT_describe_yourself']).lower()
   text_desc_2 = str(row2['TEXT_describe_yourself']).lower()
   F = 0 

   #super simple keyword search 
   values = ['christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'catholic' ]
   for key in values: 
     if (key in text_partner_desc_1 or key in text_desc_1) and (key in text_partner_desc_2 or key in text_desc_2):
        F += 5 

   traits = ['wholesome', 'fun', 'humor', 'kind', 'smart', 'caring', 'chill', 'witty', 'optimistic',
                  'pessimistic', 'introvert', 'extrovert', 'nerdy', 'curious', 'shy', 'athletic', 'active', 'studious', 'hardworking',
                  'outgoing', 'romantic', 'easy-going', 'conversation', 'analytical', 'mature', 'generous', 'spontaneous', 'artsy', 'musical', 'serious', 
             'awkward', 'quirky', '']
   for key in traits:
     if (key in text_partner_desc_1 and key in text_desc_2) or (key in text_partner_desc_2 and key in text_desc_1):
       F += 0.6
               
   interests = ['history', 'math', 'science', 'art', 'english', 'stats', 'hiking', 'climbing', 'boba', 'sports', 'music', 'piano', 'violin', 'comedy', 'psychology', 'art', 'funny', 'friend', 'relationship', 'football', 'baseball', 'basketball']
   for key in interests:
     if (key in text_partner_desc_1 and key in text_desc_2) or (key in text_partner_desc_2 and key in text_desc_1):
       F += 0.25
      
   return A + B + C + D + E + F

#input: gender balanced df 
#gale-shapley algorithm with preferences computed from compatability()
#return: pairings by id 
def matching_algo(df, hetero = True): 

    if hetero:
      df_male = df[df['gender'] == 'Male']
      df_female = df[df['gender'] == 'Female']
      assert len(df_male) == len(df_female), "error"
      N = len(df_male)
    else:
      n = len(df)
      assert n % 2 == 0, 'error'
      cut = int(n / 2)
       
      #splitting into 2 sets (labeled male/female for simplicity)
      df_male = df.iloc[:cut]
      df_female = df.iloc[cut:]
      N = cut 

    compat_matrix = [[compatibility(df_male.iloc[i], df_female.iloc[j]) for j in range(N)] for i in range(N)]
    temp_paired = [[0 for j in range(N)] for i in range(N)]
    

    preference_ordering = [[(-compat_matrix[i][j], j) for j in range(N)] for i in range(N)]
    for li in preference_ordering:
      heapq.heapify(li)
    next_to_propose = [0 for i in range (N)] #index in preference_ordering 
    curr_paired_men = []
    count = 0 
    while len(curr_paired_men) < N:
      print(f"***ROUND {count}***")
      for i in range(N):
        if i not in curr_paired_men: 
           #get first woman on unmatched person's list, this should never fail 
           w =  heapq.heappop(preference_ordering[i])[1]
           curr_match_w = -1 
           for m in range(N):
             if temp_paired[m][w]: 
               curr_match_w = m
          #if w is free
           if curr_match_w == -1:
             #match (i, w) 
             temp_paired[i][w] = 1 
             curr_paired_men.append(i) 
           #else if a pair exists 
           else: 
              #if w prefers i to her current match:
              if compat_matrix[i][w] > compat_matrix[curr_match_w][w]: 
                 #i replaces w's current match 
                 temp_paired[curr_match_w][w] = 0 
                 temp_paired[i][w] = 1 
                 curr_paired_men.remove(curr_match_w)
                 curr_paired_men.append(i) 
              #otherwise curr_match_w remains paired, i is unpaired 
      count += 1 
    
    #return matches as df, with names, emails, shared date ideas, and shared interests 
    df_matches = pd.DataFrame(columns=['name1', 'email1', 'year1', 'gender1', 'description_1', 'ideal_partner_1', 'name2', 'email2', 'year2', 'gender2', 'description_2', 'ideal_partner_2', 'shared_date_loc', 'shared_date_events', 'shared_looking_for'])

    matches = []
    #format into list of matches 
    for i in range(N):
      for j in range(N):
        if temp_paired[i][j]:
          loc_list_1 = str(df_male.iloc[i]['date_loc']).split(';')
          loc_list_2 = str(df_female.iloc[j]['date_loc']).split(';')
          common_loc = list(set(loc_list_1).intersection(loc_list_2))
          locs_available = ['A bubble tea shop', 'A cafe', 'Ice Cream']
          common_loc = list(set(loc_list_1).intersection(loc_list_2))
          if len(common_loc) == 0:
             date_loc = locs_available[randint(0,2)]
          else:
             date_loc = common_loc[randint(0, len(common_loc) -1)]
          

          looking_list_1 = str(df_male.iloc[i]['looking_for']).split(';')
          looking_list_2 = str(df_female.iloc[j]['looking_for']).split(';')
          common_looking = list(set(looking_list_1).intersection(looking_list_2))
          looking_for = ';'.join(common_looking)

          events_list_1 = str(df_male.iloc[i]['date_events']).split(';')
          events_list_2 = str(df_female.iloc[j]['date_events']).split(';')
          common_events = list(set(events_list_1).intersection(events_list_2))
          date_events = ';'.join(common_events)
          
          df_matches = df_matches.append({'name1': df_male.iloc[i]['name'],
                             'email1': df_male.iloc[i]['email'],
                             'year1': df_male.iloc[i]['year'],
                             'gender1': df_male.iloc[i]['gender'],
                             'description_1': df_male.iloc[i]['TEXT_describe_yourself'],
                             'ideal_partner_1': df_male.iloc[i]['TEXT_looking_partner'],
                             'name2': df_female.iloc[j]['name'],
                             'email2': df_female.iloc[j]['email'],
                             'year2': df_female.iloc[j]['year'],
                             'gender2': df_female.iloc[j]['gender'],
                             'description_2': df_female.iloc[j]['TEXT_describe_yourself'],
                             'ideal_partner_2': df_female.iloc[j]['TEXT_looking_partner'],
                             'shared_date_loc': date_loc,
                             'shared_date_events': date_events,
                             'shared_looking_for': looking_for}, ignore_index = True)
                            
    return df_matches



data = pd.read_csv('data.csv', delimiter = ',', index_col = False, names = ['timestamp', 'username', 'name', 'email', 'year', 'gender', 'match_gender', 'availability', 'date_loc', 'date_events', 'looking_for', 'TEXT_looking_partner', 'TEXT_describe_yourself', 'blind', 'personality', 'cottage', 'invite']).iloc[1:]
data["id"] = data.index
data

#filter out duplicates, accepting last response 
i = 0
to_drop = []
for i in range(len(data)-1, -1, -1):
    username = data.iloc[i]['email']
    for j in range(i-1, -1, -1):
      if username == data.iloc[j]['email']:
        to_drop.append(j)

data = data.drop(data.index[to_drop])

data['email'].value_counts()
data.to_csv('no_dup_data.csv')

#start here 
data = pd.read_csv('no_dup_data.csv')

#give non-hetero /non-binary to huahao, algo cannot process 
non_hetero_filter = ((data['gender'] != 'Male') & (data['gender'] != 'Female')) \
| ((data['gender'] == 'Female') & ((data['match_gender'] != 'Male') & (data['match_gender'] != 'Female')))
df_non_hetero = data[non_hetero_filter]

df_non_hetero
df_non_hetero.to_csv('huahao.csv')

#remove from pool 
data = data[~non_hetero_filter]

grad_filter = ((data['year'] != '2020') & (data['year'] != '2021') & (data['year'] != '2022') & (data['year'] != '2023'))
df_grad = data[grad_filter]
df_grad.to_csv('grad.csv')
data = data[~grad_filter]

mm_filter = ((data['gender'] == 'Male') & (data['match_gender'] == 'Male'))
ff_filter = ((data['gender'] == 'Female') & (data['match_gender'] == 'Female'))
##########
df_male_male = data[mm_filter]
df_female_female = data[ff_filter]

data = data[~(mm_filter | ff_filter)]
#remaining is all undergrad, hetero matches plus bi males (more females than males)

#filter out females who just want to make friends 
female_friend_filter = ((data['looking_for'] == 'A friend') | (data['looking_for'] == 'A short date;A friend') | (data['looking_for'] == 'A friend;A platonic love')) & (data['gender'] == 'Female')
len(data[female_friend_filter])
df_female_friends = data[female_friend_filter]
#FINAL DF 
df = data[~female_friend_filter]

#check all lengths 
print(len(df_male_male))
print(len(df_female_female))
print(len(df_female_friends))
print(len(df))

df.match_gender.value_counts()
#all male are bi / don't care 
df.gender.value_counts()

df.to_csv('data_filtered.csv')

#filter blind and not blind and split 
df_blind = df[df['blind'] == 'Yes! I love surprises']
df_notblind = df[~(df['blind'] == 'Yes! I love surprises')]
len(df_blind) + len(df_notblind)

df_blind.gender.value_counts()

df_notblind.gender.value_counts()  
#if unbalanced, move some blind into non-blind to balance (not other way around)

#move 11 males from blind to not-blind 
move_to_notblind = 11
ind = []
for i in range(len(df_blind)):
  if df_blind.iloc[i]['gender'] == 'Male':
    ind.append(i)
  if len(ind) == move_to_notblind:
    break
print(ind)
rows = df_blind.iloc[ind]
rows
df_notblind = df_notblind.append(rows)
df_blind = df_blind.drop(df_blind.index[ind])


#process myers-brigg data 
mb = {}
with open('myersbrigg.csv', encoding='utf-8-sig') as csv_file:
  csv_reader = csv.reader(csv_file)
  for row in csv_reader:
    mb[row[0]] = {}
    for i in range(1, 17):
      mb[row[0]][row[i]] = 1.0 - (i - 1) * 2.0 / 15.0 #evenly spaced between 1 and -1 



df_matches_blind = matching_algo(df_blind) 
df_matches_notblind = matching_algo(df_notblind)

df_matches_blind.to_csv('blind_matches.csv')
df_matches_notblind.to_csv('notblind_matches.csv')
