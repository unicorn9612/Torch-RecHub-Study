import pandas as pd

print("1. 读取原始 dat 文件...")
# ratings: UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', engine='python', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

# users: UserID::Gender::Age::Occupation::Zip-code
users = pd.read_csv('data/ml-1m/users.dat', sep='::', engine='python', 
                    names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')

# movies: MovieID::Title::Genres
movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', engine='python', 
                     names=['movie_id', 'title', 'genres'], encoding='latin-1')

print("2. 开始合并数据...")
# 先合并用户信息，再合并电影信息
data = ratings.merge(users, on='user_id', how='left')
data = data.merge(movies, on='movie_id', how='left')

print("3. 保存为 ml-1m.csv...")
data.to_csv('data/ml-1m/ml-1m.csv', index=False)
print("合并完成！总计行数:", len(data))