import numpy as np
import csv
import ast

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def plot_hist(data):
    plt.hist(data)
    plt.show()

def preprocess():
    overviews = []
    all_keyword_ids = []
    points = []
    all_genre_ids = []

    with open('tmdb_5000_movies.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # 0 - budget,1 - genres, 4 - keywords, 6 - title, 7 - overview, 8 - popularity, 9 - production, 10 - revenue, 18 - vote average, 19 - vote count
                h = str(eval(row[1])[0]['id']) + "," + str(eval(row[4])[0]['id']) + "," + row[6] + "," + row[7] + "," + row[8] + "," + str(eval(row[9])[0]['id']) + "," + row[12] + ',' + row[18] +',' + row[19]
                if int(row[19]) > 20 and int(row[12]) > 1000000: # vote count > 20
                    new_point = {'revenue': int(row[12]), 'popularity': float(row[18]), 'id': int(row[3]),  'production_company': int(eval(row[9])[0]['id'])}
                    points.append(new_point)
                    keyword = ' '.join([str(x['id']) for x in eval(row[4])])
                    all_keyword_ids.append(keyword)
                    genre = ' '.join([str(x['id']) for x in eval(row[1])])
                    all_genre_ids.append(genre)
                    overviews.append(row[7])
            except:
                pass

    def popularity_hist():
        popularities = []
        for point in points:
            popularities.append(point['popularity'])
        plot_hist(popularities)

    def revenue_hist():
        revenues = []
        for point in points:
            revenues.append(point['revenue'])
        plot_hist(revenues)

    def update_revenue_bins():
        counts = [0,0,0]
        for point in points:
            if point['revenue'] < 25000000:
                counts[0] += 1
                point['revenue'] = 0
            elif point['revenue'] >= 25000000 and point['revenue'] < 100000000:
                counts[1] += 1
                point['revenue'] = 1
            else:
                counts[2] += 1
                point['revenue'] = 2

    update_revenue_bins()

    def update_vote_bins():
        zeros = 0
        ones = 0
        twos = 0
        threes = 0
        fours = 0
        for point in points:
            popularity = point['popularity']
            if popularity < 5:
                point['popularity'] = 0
                zeros += 1
            elif popularity >= 5 and popularity < 6:
                point['popularity'] = 1
                ones += 1
            elif popularity >= 6 and popularity < 7:
                point['popularity'] = 2
                twos += 1
            elif popularity >= 7 and popularity < 8:
                point['popularity'] = 3
                threes += 1
            else:
                point['popularity'] = 4
                fours += 1
        # print(zeros)
        # print(ones)
        # print(twos)
        # print(threes)
        # print(fours)
    #update_vote_bins()

    def overview_bag_of_words():
        countvec = CountVectorizer(stop_words='english')
        cdf = countvec.fit_transform(overviews)
        bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names())
        bow = bow.drop(columns=bow.columns[bow.eq(0).mean()>0.99])
        bow.rename(columns=lambda x: f'overview_{x}', inplace=True)
        return bow    

    def keywords_bag_of_words():
        countvec = CountVectorizer(stop_words='english')
        cdf = countvec.fit_transform(all_keyword_ids)
        bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names())
        bow = bow.drop(columns=bow.columns[bow.eq(0).mean()>0.99])
        bow.rename(columns=lambda x: f'keywords_{x}', inplace=True)
        return bow

    def genres_bag_of_words():
        countvec = CountVectorizer(stop_words='english')
        cdf = countvec.fit_transform(all_genre_ids)
        bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names())
        # bow = bow.drop(columns=bow.columns[bow.eq(0).mean()>0.99])
        bow.rename(columns=lambda x: f'genres_{x}', inplace=True)
        return bow

    
    def combine_df():
        joined = pd.DataFrame(points)
        overview_bow = overview_bag_of_words()
        keyword_bow = keywords_bag_of_words()
        genre_bow = genres_bag_of_words()

        # joined = pd.concat([joined.reset_index(drop=True), overview_bow.reset_index(drop=True)], axis=1)
        joined = pd.concat([joined.reset_index(drop=True), keyword_bow.reset_index(drop=True)], axis=1)
        joined = pd.concat([joined.reset_index(drop=True), genre_bow.reset_index(drop=True)], axis=1)
        
        return joined

    joined_df = combine_df()

    casts = []
    directors = []
    def read_credits_csv():
        with open('tmdb_5000_credits.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    if int(row[0]) in joined_df['id'].values:
                        actors = ' '.join([str(x['id']) for x in eval(row[2])])
                        casts.append(actors)
                        crew = eval(row[3])
                        for member in crew:
                            if member['job'] == 'Director':
                                directors.append(member['id'])
                except:
                    pass
    
    
    read_credits_csv()

    def cast_bag_of_words():
        countvec = CountVectorizer()
        cdf = countvec.fit_transform(casts)
        bow = pd.DataFrame(cdf.toarray(), columns = countvec.get_feature_names())
        bow = bow.drop(columns=bow.columns[bow.eq(0).mean()>.995])
        bow.rename(columns=lambda x: f'cast_{x}', inplace=True)
        return bow

    def join_cast_bow():
        cast_bow = cast_bag_of_words()
        joined = pd.concat([joined_df.reset_index(drop=True), cast_bow.reset_index(drop=True)], axis=1)
        return joined

    joined_df = join_cast_bow()

    def one_hot_encoding():
        return pd.get_dummies(joined_df, columns=["production_company"], prefix=["production_company"])
        
    return one_hot_encoding()

def convert_np():
    df = preprocess()
    print(df)
    # drop('id', axis=1)
    ids = list(df.pop('id'))
    # sims = cosine_similarity(df)
    # np.fill_diagonal(sims, .5)
    # count = 0
    # for l in sims:
    #     if list(ids)[count] == 579:
    #         print(str(list(ids)[count]) + ' min similarity:' + str(min(list(l))) + ', average similarity: ' + str(sum(l)/len(l)) + ', max similarity: ' +  str(max(list(l))) + ", max index: " + str(ids[list(l).index(max(l))]))
    #     count += 1
        
    # print(sims.max())
    data = df.to_numpy()
    return data

if __name__ == '__main__':
    # preprocess()
    print(convert_np())