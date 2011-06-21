from joblib import delayed, Parallel
import mtest


NCASES = 100000
NJOBS = -1
NS = range(3, 20) + range(20, 100, 10)
SGMS = [0.25, 0.5, 1.0, 1.5, 2.0]

if __name__ == '__main__':
    Parallel(n_jobs=NJOBS, verbose=1)(delayed(mtest.typeI_table)(n, n, NCASES)
                                      for n in NS)
    Parallel(n_jobs=NJOBS, verbose=1)(delayed(mtest.typeII_table)(n, n, NCASES,
                                                                  1.0, std)
                                      for n in NS for std in SGMS)
