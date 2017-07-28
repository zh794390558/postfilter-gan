from  __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import argparse
import logging
import subprocess

try:
    from tqdm import tqdm
except ImportError as e:
    subprocess.check_call('./install.sh')
    from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG
                    )

def main(FLAGS):
    if not os.path.exists(FLAGS.wav_dir):
        os.mkdir(FLAGS.wav_dir)

    features = os.listdir(FLAGS.feature_dir)
    f0s = [ os.path.join(FLAGS.f0_dir, os.path.splitext(f)[0] + '.f0') for f in features]

    for f in f0s:
        if not os.path.exists(f):
            raise ValueError('featues corespond f0 ({}) file not exsits'.format(f))

    qbar = tqdm(enumerate(zip(features, f0s)))
    for i, (feature, f0) in qbar:
        filename = os.path.basename(f0)
        filename = os.path.splitext(filename)[0] + '.wav'
        wavfile = os.path.join(FLAGS.wav_dir, filename)
        wavfile = os.path.abspath(wavfile)

        feature = os.path.join(FLAGS.feature_dir, feature)
        feature = os.path.abspath(feature)

        qbar.set_description('Process {}'.format(wavfile))

        cmd = os.path.abspath(os.path.join(FLAGS.tool_dir,'straight_mceplsf'))
        args = ' -f 22050 -lsf -order 40 -shift 5 -f0file {} -syn {} {}'.format(f0, feature, wavfile)

        subprocess.check_call(cmd+args, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate wav from feature and F0')
    parser.add_argument('--feature_dir', type=str, default='data/lsf', help='File containing the feaures')
    parser.add_argument('--f0_dir', type=str, default='/gfs/atlastts/StandFemale_22K/nature/postf0/', help='File containing the F0')
    parser.add_argument('--wav_dir', type=str, default='data/wav', help='File containing the output of generate wave files')
    parser.add_argument('--tool_dir', type=str, default='tools', help='File containing the vocoder tools')

    FLAGS = parser.parse_args()
    logging.info(FLAGS)

    main(FLAGS)
