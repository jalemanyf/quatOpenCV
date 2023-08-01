from stare.stare_detector2 import StareDetector
from chrono.chrono_counter import ChronoCounter
from people.people_detector import PeopleDetector
from datetime import datetime, timedelta
from itertools import chain
import cv2
from pythonosc import udp_client


def main():
    stare_detector = StareDetector()
    people_detector = PeopleDetector()
    chrono_counter = ChronoCounter()
    cap = cv2.VideoCapture(0)

    # OSC client setup
    osc_ip = "192.168.1.33"
    osc_port = 7001
    osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)

    while True:
        current_time = datetime.now()
        date_string = current_time.strftime("%Y-%m-%d")
        time_string = current_time.strftime("%H:%M:%S")
        ret, frame = cap.read()
        if ret:
            stare_data = stare_detector.check_frame(frame)
            people_data = people_detector.check_frame(frame)
            timestamped_ids = chrono_counter.check_ids([fi['face_id'] for fi in stare_data['faces_info']])
            timestamped_ids = [
                tid
                for tid in timestamped_ids
                if (timestamped_ids[tid]['last'] - timestamped_ids[tid]['start']) > timedelta(5, 0)
            ]

            for fi in stare_data['faces_info']:
                output_list = [
                    date_string,
                    time_string,
                    len(people_data),
                    stare_data['unique_embeddings'],
                    fi['face_id'],
                    fi['dominant_emotion'],
                    *fi['emotions'].values(),
                    fi['dominant_race'],
                    fi['gender'],
                    *fi['gender_prob'].values(),
                    fi['age'],
                ]
                print(f"{output_list = }")

                # OSC
                osc_client.send_message("/Metrics/DeepFace", output_list)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()