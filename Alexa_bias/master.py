import subprocess
import time

def run_slave():
    while True:
        process = subprocess.Popen(["python", "does_not_understand.py"])
    
        while True:
            time.sleep(1)  # Check every 1 second

            # Read the status file
            with open("status.txt", "r") as f:
                status = f.read()

            # If the status indicates to stop, terminate the process
            if status == '5':
                print("Terminating does_not_understand.py")
                process.terminate()
                process.kill()
                process.kill()
                process.kill()
                process.wait()
                runnning=0

                with open("status.txt", "w") as f:
                    f.write(str(runnning))
                    
                break

if __name__ == "__main__":
    run_slave()
