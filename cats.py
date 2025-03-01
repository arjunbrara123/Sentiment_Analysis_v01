# Create config and colour sets
emotion_colours = {
    'joy': 'green',
    'neutral': 'yellow',
    'surprise': 'grey',
    'sadness': 'pink',
    'anger': 'red',
    'disgust': 'maroon',
    'fear': 'orange'
}
emotion_weights = {
    'joy': 1.5,
    'neutral': -0.5,
    'sadness': -1.0,
    'anger': -2.0,
    'disgust': -2.0,
    'fear': 0,
    'surprise': 0,
}
emotion_categories = list(emotion_colours.keys())
# product_colours = {
#     'Energy': 'limegreen',
#     'Appliance Repair': 'darkcyan',
#     'Gas Products': 'dodgerblue',
#     'Plumbing & Drains': 'peru',
#     'Unknown': 'grey'
# }
product_colours = {
    'Energy': 'orange',
    'Appliance Repair': 'maroon',
    'Home Electrical': 'maroon',
    'Gas Products': 'green',
    'Plumbing & Drains': 'magenta',
    'Unknown': 'grey'
}
company_emoji_map = {
    "British Gas": "🌎 British Gas",
    "HomeServe": "🧮 HomeServe",
    "CheckATrade": "🧮 CheckATrade",
    "Domestic & General": "🧮 Domestic & General",
    "Corgi HomePlan": "🧮 Corgi HomePlan",
    "247 Home Rescue": "🧮 247 Home Rescue",
    "Octopus": "⚡ Octopus",
    "OVO": "⚡ OVO",
}
product_emoji_map = {
    "All": "🌎 All",
    "Gas Products": "🚿 Gas Products",
    "Energy": "⚡ Energy",
    "Plumbing & Drains": "🪠 Plumbing & Drains",
    "Appliance Cover": "📺 Appliance Cover",
    "Home Electrical": "🔦 Home Electrical",
    "Heating": "🔥 Heating",
    "Pest Control": "🐀 Pest Control",
    "Unknown": "🃏 Unknown",
}
product_categories = list(product_colours.keys())
energy_colours = {
    'British Gas': 'blue',
    'Eon Energy': 'maroon',
    'Eon Next': 'red',
    'Octopus': 'pink',
    'OVO': 'green',
    'Scottish Power': 'lime',
    'Unknown': 'grey'
}
insurer_colours = {
    'British Gas': 'blue',
    'HomeServe': 'red',
    'Domestic & General': 'darkcyan',
    'Corgi HomePlan': 'orange',
    '247 Home Rescue': 'brown',
    'CheckATrade': 'grey'
}
aspects_map = {
    "Appointment Scheduling": "⌚ Appointment Scheduling",
    "Customer Service": "📞 Customer Service",
    "Response Speed": "🥇 Response Speed",
    "Engineer Experience": "🧑‍🔧 Engineer Experience",
    "Solution Quality": "🧠 Solution Quality",
    "Value For Money": "💵 Value For Money",
}
aspects = list(aspects_map.keys())
aspect_colours = {
    "Appointment Scheduling": "#6a76ac", #darkblue
    "Customer Service": "#da3d34", #darkred
    "Response Speed": "#ed9f40", #orange
    "Engineer Experience": "#00c2e0", #lightblue
    "Solution Quality": "#ff78cb", #pink
    "Value For Money": "#00a480", #green
}
