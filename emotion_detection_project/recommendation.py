# recommendation.py

# Define emotion-based recommendations
emotion_recommendations = {
    "Angry": {
        "quote": "“For every minute you are angry you lose sixty seconds of happiness.” - Ralph Waldo Emerson",
        "song": "Imagine Dragons - Believer",
        "activity": "Take deep breaths and try a quick walk outside to cool off.",
        "music_link": "https://www.youtube.com/watch?v=7wtfhZwyrcc"
    },
    "Disgust": {
        "quote": "“Disgust is the mother of all moral emotions.” - Peter Singer",
        "song": "Coldplay - Fix You",
        "activity": "Reflect on what caused the disgust and try to change your environment.",
        "music_link": "https://www.youtube.com/watch?v=YQHsXMglC9A"
    },
    "Fear": {
        "quote": "“The only thing we have to fear is fear itself.” - Franklin D. Roosevelt",
        "song": "Lindsey Stirling - Crystallize",
        "activity": "Take a few moments to breathe and think of positive outcomes.",
        "music_link": "https://www.youtube.com/watch?v=aHjpOzsQ9YI"
    },
    "Happy": {
        "quote": "“Happiness depends upon ourselves.” - Aristotle",
        "song": "Pharrell Williams - Happy",
        "activity": "Share your happiness with others, spread positivity.",
        "music_link": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
    },
    "Sad": {
        "quote": "“Tears are words that need to be written.” - Paulo Coelho",
        "song": "Adele - Someone Like You",
        "activity": "Talk to a friend or take some time to reflect and relax.",
        "music_link": "https://www.youtube.com/watch?v=hLQl3WQQoQ0"
    },
    "Surprise": {
        "quote": "“Life is full of surprises, but you need to be open to them.” - Unknown",
        "song": "Queen - Don't Stop Me Now",
        "activity": "Embrace the surprise and go with the flow.",
        "music_link": "https://www.youtube.com/watch?v=2xW3WsS0F6g"
    },
    "Neutral": {
        "quote": "“Just because you don’t feel anything, doesn’t mean you’re numb.” - Unknown",
        "song": "Lofi Hip Hop for studying",
        "activity": "Relax, take a break, or enjoy something creative.",
        "music_link": "https://www.youtube.com/watch?v=5qap5aO4i9A"
    }
}

# Function to get recommendations based on emotion
def get_recommendations(emotion):
    return emotion_recommendations.get(emotion, None)
