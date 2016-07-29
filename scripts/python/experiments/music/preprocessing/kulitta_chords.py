import os


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
MUSIC_DATA_ROOT = os.path.join(HAMLET_ROOT, 'data/data/music')
KULITTA_DATA_ROOT = os.path.join(MUSIC_DATA_ROOT, 'kulitta')
KULITTA_chord1_DATA_ROOT = os.path.join(KULITTA_DATA_ROOT, 'KulittaData_chord1_20160710')

# print os.listdir(KULITTA_chord1_DATA_ROOT)


def read_kulitta_chord1(data_root):
    print os.listdir(HAMLET_ROOT)
    print os.listdir(MUSIC_DATA_ROOT)
    print os.listdir(KULITTA_DATA_ROOT)
    print os.listdir(data_root)

read_kulitta_chord1(KULITTA_chord1_DATA_ROOT)
