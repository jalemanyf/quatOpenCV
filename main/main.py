from stare.stare_detector2 import StareDetector
from chrono.chrono_counter import ChronoCounter
from people.people_detector import PeopleDetector
#from camera.camer_handler import CameraHandler
from datetime import datetime, timedelta
import cv2


def main():
    stare_detector = StareDetector()
    people_detector = PeopleDetector()
    chrono_counter = ChronoCounter()
    cap = cv2.VideoCapture(0)
    #with CameraHandler() as cap:
    while True:
        current_time = datetime.now()
        date_string = current_time.strftime("%Y-%m-%d")
        time_string = current_time.strftime("%H:%M:%S")
        ret, frame = cap.read()
        if ret:
            stare_data = stare_detector.check_frame(frame)
            people_data = people_detector.check_frame(frame)
            timestamped_ids = chrono_counter.check_ids([ fi['face_id'] for fi in stare_data['faces_info'] ])
            timestamped_ids = [
                tid
                for tid in timestamped_ids
                if (timestamped_ids[tid]['last'] - timestamped_ids[tid]['start']) > timedelta(5, 0)
            ]
            output_list = [
                [
                    date_string,
                    time_string,
                    len(people_data),
                    stare_data[ 'unique_embeddings'],
                    fi['face_id'],
                    fi['dominant_emotion'],
                    fi['emotions'],
                    fi['dominant_race'],
                    fi['gender'],
                    fi['gender_prob'],
                    fi['age']
                ]
                for fi in stare_data['faces_info']
            ]
            print(f"{output_list = }")
            for fi in stare_data['faces_info']:
                cv2.rectangle(frame, (fi['pos']['left'], fi['pos']['top']), (fi['pos']['right'], fi['pos']['bottom']),  (255,0,255), 2)
                cv2.putText(frame, f"Face ID: {fi['face_id']}", (fi['pos']['left'], fi['pos']['top'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Emotion: {fi['dominant_emotion']}", (fi['pos']['right'], fi['pos']['top'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Age: {fi['age']}", (fi['pos']['right'], fi['pos']['top']+ 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255.0), 2)
                cv2.putText(frame, f"Gender: {fi['gender']}", (fi['pos']['right'], fi['pos']['top'] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print(f"frames per second: {1.0/(datetime.now() - current_time).total_seconds()}")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

