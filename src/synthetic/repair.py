# New code Tim
# Function to parse and recover JSON data
# TODO: merge this with main code
def parse_json(data):
    try:
        return json_repair.loads(data), True
    except Exception as e:
        return data, False


# Process the dataframe
successful_json, failed_json = [], []
for row in df.itertuples():
    json_data, success = parse_json(row[1])
    (successful_json if success else failed_json).append(json_data)

salvaged_tweets = []
for json in successful_json:
    if type(json) == dict:
        for tweet in json.values():
            salvaged_tweets.append(tweet)
    elif type(json) == list:
        if type(json[0]) == dict:
            for d in json:
                for key in d.keys():
                    if key == "text" or key == "tweet" or key == "content":
                        salvaged_tweets.append(d[key])
        else:
            salvaged_tweets.extend(json)
