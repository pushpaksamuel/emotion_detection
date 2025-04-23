from image_emotion import process_image
from video_emotion import process_video

def main():
    print("\n🎭 Emotion Detection System")
    print("============================")
    print("Choose input type:")
    print("1. Image 🖼️")
    print("2. Video (20 sec) 🎥")

    choice = input("\nEnter your choice (1/2): ")

    if choice == '1':
        img_path = input("Enter the path to the image: ")
        process_image(img_path)
    elif choice == '2':
        print("\n[INFO] Starting real-time video emotion detection... Press 'q' to quit early.")
        process_video()
    else:
        print("\n[ERROR] Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
