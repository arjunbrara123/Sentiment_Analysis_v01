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
product_colours = {
    'Energy': 'orange',
    'Appliance Repair': 'maroon',
    'Home Electrical': 'maroon',
    'Gas Products': 'green',
    'Plumbing & Drains': 'magenta',
    'Unknown': 'grey'
}
product_emoji_map = {
    "Gas Products": "🚿 Gas Products",
    "Energy": "⚡ Energy",
    "Plumbing & Drains": "🪠 Plumbing & Drains",
    "Appliance Repair": "📺 Appliance Repair",
    "Home Electrical": "🔦 Home Electrical",
    "Heating": "🔥 Heating",
    "Unknown": "🃏 Unknown"
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
    'Domestic and General': 'green',
    'Corgi HomePlan': 'orange',
    '247 Home Rescue': 'brown'
}
aspects_map = {
    "Appointment Scheduling": "⌚ Appointment Scheduling",
    "Customer Service": "☎️ Customer Service",
    "Response Speed": "🎽 Response Speed",
    "Engineer Experience": "🧑‍🔧 Engineer Experience",
    "Solution Quality": "🥇 Solution Quality",
    "Value For Money": "💶 Value For Money",
}
aspects = list(aspects_map.keys())
