from enum import IntFlag, auto

from numpy import number


class Selection(IntFlag):
    @classmethod
    def max_value(cls):
        return 2 ** len(cls) - 1

    def __invert__(self):
        return self.__class__(self.__class__.max_value() - int(self))

    def __format__(self, format_spec):
        return str(self)

    @classmethod
    def create(cls, value):
        # can't put this inside _missing_ since _missing_ must return a Selection object
        if value is None:
            return None
        return cls(value)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            if value == "any":
                return ~cls(0)
            else:
                res = cls(0)
                for sel in value.split(','):
                    res |= cls[sel.strip()]
                return res
        elif isinstance(value, number):
            return cls(value.item())
        elif isinstance(value, int) and value >= 0:
            if value <= cls.max_value():
                return super()._missing_(value)
            else:
                raise ValueError(f"{cls.__name__}: {value} is out of range ({cls.max_value()}).")
        elif isinstance(value, cls):
            return value
        else:
            raise ValueError(f"{cls.__name__}: {value} is invalid. Valid selections: {cls._member_names_}")

# selections from O2Physics::daily-20260302-0000
class EventSel(Selection):
    sel8 = auto()
    sel7 = auto()
    selKINT7 = auto()
    selTVX = auto()
    selNoTimeFrameBorder = auto()
    selNoITSROFrameBorder = auto()
    selNoSameBunchPileup = auto()
    selIsGoodZvtxFT0vsPV = auto()
    selNoCollInTimeRangeStandard = auto()
    selNoCollInRofStandard = auto()
    selUpcSingleGapA = auto()
    selUpcSingleGapC = auto()
    selUpcDoubleGap = auto()
    sel8Full = sel8 | selNoSameBunchPileup
    sel8FullPbPb = sel8 | selNoCollInTimeRangeStandard | selNoCollInRofStandard
    selUnanchoredMC = selTVX
    selMC = selTVX | selNoTimeFrameBorder
    selMCFull = selTVX | selNoTimeFrameBorder | selNoSameBunchPileup
    selMCFullPbPb = selTVX | selNoCollInTimeRangeStandard | selNoCollInRofStandard
    sel7KINT7 = sel7 | selKINT7

# selections from O2Physics::daily-20260302-0000
class TrigSel(Selection):
    noTrigSel = auto()
    JetChLowPt = auto()
    JetChHighPt = auto()
    TrackLowPt = auto()
    TrackHighPt = auto()
    JetD0ChLowPt = auto()
    JetD0ChHighPt = auto()
    JetLcChLowPt = auto()
    JetLcChHighPt = auto()
    EMCALReadout = auto()
    JetFullHighPt = auto()
    JetFullLowPt = auto()
    JetNeutralHighPt = auto()
    JetNeutralLowPt = auto()
    GammaVeryHighPtEMCAL = auto()
    GammaVeryHighPtDCAL = auto()
    GammaHighPtEMCAL = auto()
    GammaHighPtDCAL = auto()
    GammaLowPtEMCAL = auto()
    GammaLowPtDCAL = auto()
    GammaVeryLowPtEMCAL = auto()
    GammaVeryLowPtDCAL = auto()

# selections from O2Physics::daily-20260302-0000
class RCTSel(Selection):
    CPVBad = auto()
    EMCBad = auto()
    EMCLimAccMCRepr = auto()
    FDDBad = auto()
    FT0Bad = auto()
    FV0Bad = auto()
    HMPBad = auto()
    ITSBad = auto()
    ITSLimAccMCRepr = auto()
    MCHBad = auto()
    MCHLimAccMCRepr = auto()
    MFTBad = auto()
    MFTLimAccMCRepr = auto()
    MIDBad = auto()
    MIDLimAccMCRepr = auto()
    PHSBad = auto()
    TOFBad = auto()
    TOFLimAccMCRepr = auto()
    TPCBadTracking = auto()
    TPCBadPID = auto()
    TPCLimAccMCRepr = auto()
    TRDBad = auto()
    ZDCBad = auto()
    NRCTSelectionFlags = auto()
    Dummy24 = auto()
    Dummy25 = auto()
    Dummy26 = auto()
    Dummy27 = auto()
    Dummy28 = auto()
    Dummy29 = auto()
    Dummy30 = auto()
    CcdbObjectLoaded = auto()
    CBT = FT0Bad | ITSBad | TPCBadTracking | TPCBadPID
    CBT_LAMCRBad = CBT | ITSLimAccMCRepr | TPCLimAccMCRepr
    CBT_hadronPID = CBT | TOFBad
    CBT_hadronPID_LAMCRBad = CBT_hadronPID | ITSLimAccMCRepr | TPCLimAccMCRepr | TOFLimAccMCRepr
    CBT_calo = CBT | EMCBad
    CBT_calo_LAMCRBad = CBT_calo | ITSLimAccMCRepr | TPCLimAccMCRepr | EMCLimAccMCRepr

# selections from O2Physics::daily-20260302-0000
class TrackSel(Selection):
    trackSign = auto() # 1 = positive, 0 = negative
    globalTrack = auto()
    qualityTrack = auto()
    qualityTrackWDCA = auto()
    hybridTrack = auto()
    notBadMcTrack = auto()
    embeddedTrack = auto()
