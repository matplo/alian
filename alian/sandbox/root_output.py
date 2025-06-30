
import ROOT

from heppyy.util.logger import Logger
log = Logger()

# singleton class for for analysis output root file
class SingleRootFile(object):

    __instance = None

    def __new__(cls, fname = 'output.root'):
        if SingleRootFile.__instance is None:
            SingleRootFile.__instance = object.__new__(cls)
            SingleRootFile.__instance.filename = fname
            SingleRootFile.__instance.root_file = ROOT.TFile(fname, 'RECREATE')            
            SingleRootFile.__instance.objects = []
            log.info(f'new root file: {SingleRootFile.__instance.root_file.GetName()}')
        return SingleRootFile.__instance

    def add(self, obj):
        if obj:
            self.objects.append(obj)

    def __del__(self):
        if SingleRootFile.__instance:
            if SingleRootFile.__instance.root_file.IsOpen():
                log.info(f'closing {SingleRootFile.__instance.root_file.GetName()}')
                SingleRootFile.__instance.close()
        
    def close(self):
        _rfile = self.root_file
        log.info(f'SingleRootFile: root_file is {_rfile}')
        if _rfile:
            _rfile.cd()
            _ = [o.Write() for o in self.objects]
            _rfile.Write()
            _rfile.Close()
            log.info(f'SingleRootFile: wrote {_rfile.GetName()}')
            log.info(f'SingleRootFile: purging {_rfile.GetName()}')
            _rfile = ROOT.TFile(self.filename, 'UPDATE')
            _rfile.Purge()
            _rfile.Write()
            _rfile.Close()
        else:
            log.debug('SingleRootFile: no root file to write')
        _rfile = None
        log.info(f'done {self.filename}')
