import sounddevice as sd

def list_devices():
    print("Listing Audio Devices...")
    try:
        devices = sd.query_devices()
        print(devices)
        
        default_input = sd.query_devices(kind='input')
        print(f"\nDefault Input Device:\n{default_input}")
        
    except Exception as e:
        print(f"❌ Error querying devices: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure a microphone is connected.")
        print("2. Check Windows Privacy settings for Microphone access.")
        print("3. Try running this script as Administrator (sometimes required).")
        print("4. Reinstall drivers if necessary.")

if __name__ == "__main__":
    list_devices()
