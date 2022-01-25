import laboratoire
import app

if __name__ == '__main__':
    print("Test script")

    to_run = 8

    if to_run == 1:
        laboratoire.num_1()
    elif to_run == 2:
        laboratoire.num_2()
    elif to_run == 3:
        laboratoire.num_3()
    elif to_run == 4:
        laboratoire.num_4()
    elif to_run == 5:
        app.rehaussement_du_signal('sound_files\\hel_fr2.wav')
    elif to_run == 7:
        app.conventional_noise_feedback_coding('sound_files\\hel_fr1.wav')
    elif to_run == 8:
        app.qmf('sound_files\\hel_fr1.wav')
    else:
        print("Not running anything")
