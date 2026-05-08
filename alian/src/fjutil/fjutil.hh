// fjutil.hh
#ifndef HEPPYY_FJUTIL_HH
#define HEPPYY_FJUTIL_HH

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <vector>
#include <fastjet/PseudoJet.hh>
#include <numpy/arrayobject.h>
#include <stdexcept>

namespace alian
{
	void process_numpy_array(PyObject *array);

	// Static initializer class
	class NumpyInitializer
	{
	public:
		NumpyInitializer()
		{
			if (_do_init() != 0)
			{
				throw std::runtime_error("NumPy initialization failed");
			}
		}

	private:
		int _do_init()
		{
			if (_import_array() < 0)
			{
				PyErr_Print();
				PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
				return -1;
			}
			return 0;
		}
	};

	// Static instance to trigger initialization
	static NumpyInitializer numpy_initializer;

	std::vector<fastjet::PseudoJet> numpy_pxpypz_to_pseudojets(PyObject *px, PyObject *py, PyObject *pz, double m, int index_offset = 0);
	std::vector<fastjet::PseudoJet> numpy_ptetaphi_to_pseudojets(PyObject *pt, PyObject *eta, PyObject *phi, double m, int index_offset = 0);

	constexpr double PION_MASS = 0.1395704;

	std::vector<fastjet::PseudoJet> numpy_pxpypz_to_tracks(PyObject *px, PyObject *py, PyObject *pz, PyObject *tracksel, int index_offset);
	std::vector<fastjet::PseudoJet> numpy_ptetaphi_to_tracks(PyObject *pt, PyObject *eta, PyObject *phi, PyObject *tracksel, int index_offset);

	class TrackInfo : public fastjet::PseudoJet::UserInfoBase
	{
	public:
		TrackInfo(short q, uint16_t track_sel) : _q(q), _track_sel(track_sel) {}

		const short q() const { return _q; }
		const uint16_t track_sel() const { return _track_sel; }

	private:
		short _q;
		uint16_t _track_sel;
	};

	class Cluster : public fastjet::PseudoJet
	{
	public:
		Cluster(
			const double px_in,
			const double py_in,
			const double pz_in,
			const double E_in,
			const float m02,
			const float m20,
			const int ncells,
			const float time,
			const bool exoticity,
			const float dbc,
			const int nlm,
			const int defn,
			const int matchedTrackN,
			PyObject* matchedTrackDeltaEta,
			PyObject* matchedTrackDeltaPhi,
			PyObject* matchedTrackP,
			PyObject* matchedTrackPt,
			PyObject* matchedTrackSel
		) : fastjet::PseudoJet(px_in, py_in, pz_in, E_in),
		_m02(m02),
		_m20(m20),
		_ncells(ncells),
		_time(time),
		_exoticity(exoticity),
		_dbc(dbc),
		_nlm(nlm),
		_defn(defn),
		_matchedTrackN(matchedTrackN),
		_matchedTrackDeltaEta(matchedTrackDeltaEta),
		_matchedTrackDeltaPhi(matchedTrackDeltaPhi),
		_matchedTrackP(matchedTrackP),
		_matchedTrackPt(matchedTrackPt),
		_matchedTrackSel(matchedTrackSel)
		{
			Py_INCREF(_matchedTrackDeltaEta);
			Py_INCREF(_matchedTrackDeltaPhi);
			Py_INCREF(_matchedTrackP);
			Py_INCREF(_matchedTrackPt);
			Py_INCREF(_matchedTrackSel);
		}

		~Cluster() {
			Py_DECREF(_matchedTrackDeltaEta);
			Py_DECREF(_matchedTrackDeltaPhi);
			Py_DECREF(_matchedTrackP);
			Py_DECREF(_matchedTrackPt);
			Py_DECREF(_matchedTrackSel);
		}
		template <typename T>
		const double delta_eta(const T &other) const { return other.eta() - eta(); };
		template <typename T>
		const double abs_delta_eta(const T &other) const { return fabs(other.eta() - eta()); };
		template <typename T>
		const double abs_delta_phi(const T &other) const { return fabs(delta_phi_to(other)); };
		template <typename T>
		const double dR(const T &other) const {
			double dphi = delta_phi_to(other);
			double deta = delta_eta(other);
			return sqrt(dphi*dphi + deta*deta);
		};
		inline const auto energy() const { return e(); };
		inline const auto m02() const { return _m02; };
		inline const auto m20() const { return _m20; };
		inline const auto ncells() const { return _ncells; };
		inline const auto time() const { return _time; };
		inline const auto exoticity() const { return _exoticity; };
		inline const auto dbc() const { return _dbc; };
		inline const auto nlm() const { return _nlm; };
		inline const auto defn() const { return _defn; };
		inline const auto matchedTrackN() const { return _matchedTrackN; };
		inline auto matchedTrackDeltaEta() const { Py_INCREF(_matchedTrackDeltaEta); return _matchedTrackDeltaEta; };
		inline auto matchedTrackDeltaPhi() const { Py_INCREF(_matchedTrackDeltaPhi); return _matchedTrackDeltaPhi; };
		inline auto matchedTrackP() const { Py_INCREF(_matchedTrackP); return _matchedTrackP; };
		inline auto matchedTrackPt() const { Py_INCREF(_matchedTrackPt); return _matchedTrackPt; };
		inline auto matchedTrackSel() const { Py_INCREF(_matchedTrackSel); return _matchedTrackSel; };
		// Copy constructor
		// Cluster(const Cluster& other)
		// 	: _m02(other.m02()),
		// 	_m20(other.m20()),
		// 	_ncells(other.ncells()),
		// 	_time(other.time()),
		// 	_exoticity(other.exoticity()),
		// 	_dbc(other.dbc()),
		// 	_nlm(other.nlm()),
		// 	_defn(other.defn()),
		// 	_matchedTrackN(other.matchedTrackN()),
		// 	_matchedTrackEta(other.matchedTrackEta()),
		// 	_matchedTrackPhi(other.matchedTrackPhi()),
		// 	_matchedTrackP(other.matchedTrackP()) {
		// 		Py_DECREF(_matchedTrackEta);
		// 		// Py_DECREF(_matchedTrackEta);
		// 		Py_DECREF(_matchedTrackPhi);
		// 		Py_DECREF(_matchedTrackP);
		// 	}

		private:
			float _m02;
			float _m20;
			int _ncells;
			float _time;
			bool _exoticity;
			float _dbc;
			int _nlm;
			int _defn;
			int _matchedTrackN;
			PyObject* _matchedTrackDeltaEta;
			PyObject* _matchedTrackDeltaPhi;
			PyObject* _matchedTrackP;
			PyObject* _matchedTrackPt;
			PyObject* _matchedTrackSel;
	};

	std::vector<Cluster> numpy_energyetaphi_to_clusters(
		PyObject *energy, PyObject *eta, PyObject *phi,
		PyObject *m02,
		PyObject *m20,
		PyObject *ncells,
		PyObject *time,
		PyObject *exoticity,
		PyObject *dbc,
		PyObject *nlm,
		PyObject *defn,
		PyObject *matchedTrackN,
		PyObject *matchedTrackDeltaEta,
		PyObject *matchedTrackDeltaPhi,
		PyObject *matchedTrackP,
		PyObject *matchedTrackPt,
		PyObject *matchedTrackSel,
		int index_offset
	);
}

#endif