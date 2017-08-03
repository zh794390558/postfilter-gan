from  __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import argparse
import logging
import subprocess
import multiprocessing
import time

try:
    from tqdm import tqdm
except ImportError as e:
    subprocess.check_call('./install.sh')
    from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG
                    )

# use `starght_mceplsf` tool to gen melcep
def gen_mcep(f0file, feature, wavfile):
    '''
    f0file: input
    wavfile: input
    feature: output
    '''
    tool = os.path.abspath(os.path.join(FLAGS.tool_dir,'straight_mceplsf'))
    args = ' -f 22050 -shift 5 -mcep -pow -order 40 -f0file {} -ana {} {}'.format(f0file, wavfile, feature)
    cmd = tool + args

    subprocess.check_call(cmd, shell=True)

def main(FLAGS):
    if not os.path.exists(FLAGS.wav_dir):
        os.mkdir(FLAGS.wav_dir)

    # syn files and f0 files
    wavs = os.listdir(FLAGS.wav_dir)
    f0s = [ os.path.join(FLAGS.f0_dir, os.path.splitext(f)[0] + '.f0') for f in wavs]

    if not os.path.exists(FLAGS.feature_dir):
        os.makedirs(FLAGS.feature_dir)

    for f in f0s:
        if not os.path.exists(f):
            raise ValueError('featues corespond f0 ({}) file not exsits'.format(f))

    start = time.time()

    processes = []
    qbar = tqdm(enumerate(zip(wavs, f0s)), total=len(f0s))
    for i, (wav, f0) in qbar:
        outfilename = os.path.basename(f0)
        outfilename = os.path.splitext(outfilename)[0] + '.' + FLAGS.feature_suffix

        wavfile = os.path.join(FLAGS.wav_dir, wav)
        wavfile = os.path.abspath(wavfile)

        feature = os.path.join(FLAGS.feature_dir, outfilename)
        feature = os.path.abspath(feature)

        qbar.set_description('Process {}'.format(wavfile))

        # multi-process
        p = multiprocessing.Process(target=gen_mcep, args=(f0, feature, wavfile))
        processes.append(p)
        p.start()

    # wait childs
    for p in processes:
        p.join()

    logging.info('Process time = {}s'.format(time.time() - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate wav from feature and F0')
    parser.add_argument('--feature_dir', type=str, default='data/cep', help='Output: File containing the feaures')
    parser.add_argument('--feature_suffix', type=str, default='cep', help='feature suffix') # cep for gen, mecp for nature
    parser.add_argument('--f0_dir', type=str, default='/gfs/atlastts/StandFemale_22K/nature/postf0/', help='File containing the F0')
    parser.add_argument('--wav_dir', type=str, default='data/wav', help='File containing the wave files')
    parser.add_argument('--tool_dir', type=str, default='tools', help='File containing the vocoder tools')

    FLAGS = parser.parse_args()
    logging.info(FLAGS)

    main(FLAGS)

