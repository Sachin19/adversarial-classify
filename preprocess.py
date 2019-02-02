labeldict = {}
for country in os.listdir("."):
    countryname = country.split(".")[1]
    countryusers = []
    if countryname in ["UK","US","Australia","Ireland","NewZealand"]:
        label = "English"
    elif countryname in ["Germany","Austria"]:
        label = "German"
    elif countryname in ["Spain","Mexico"]:
        label = "Spanish"
    else:
        label = countryname
    labeldict[countryname] = label
    for user in os.listdir(country):
        userdata = []
        for chunk in os.listdir(country+"/"+user):
            userdata.append(countryname+"/"+user+"/"+chunk)
        countryusers.append(userdata)
    if label not in users:
        users[label] = []
    users[label] += countryusers


users_data = {}
for country, value in users.items():
    random.shuffle(value)
    users_data[country] = value[:104]

for country, value in users_data.items():
    random.shuffle(value)
    users_test_data[country] = value[:11]
    users_dev_data[country] = value[11:14]
    users_train_data[country] = value[14:104]

for country, value in users_test_data.items():
    for user in value:
        random.shuffle(user)
        if len(user)>3:
            users_test_chunks += user[:3]
        else:
            users_test_chunks += user
