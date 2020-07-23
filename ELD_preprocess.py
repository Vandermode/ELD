import os
from os.path import join


def standardize_name(datadir, fn, i, prefix='IMG', dry_run=True):    
    _, ext = os.path.splitext(fn)
    src = join(datadir, fn)
    
    dst = join(datadir, '{}_{:04d}{}'.format(prefix, i,ext))

    if src == dst:
        return
    
    print('rename {} to {}'.format(src, dst))
    if not dry_run:
        os.rename(src, dst)


if __name__ == '__main__':
    cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
    suffixes = ['.CR2', '.CR2', '.nef', '.ARW']
    
    basedir = 'E:\\Vac\\Dataset\\ELD'  # modify it to your ELD dataset path

    for camera, suffix in zip(cameras, suffixes):

        for i in range(1,10+1):
            scene = 'scene-{}'.format(i)
            print('---------- {} ----------'.format(scene))
            datadir = os.path.join(basedir, camera, scene)
            
            for s in [suffix, '.JPG', '.jpg']:
                fns = sorted([fn for fn in os.listdir(datadir) if fn.endswith(s)])
                
                for i, fn in enumerate(fns, 1):
                    # dry_run first 
                    standardize_name(datadir, fn, i, dry_run=True)
                    # standardize_name(datadir, fn, i, dry_run=False)
