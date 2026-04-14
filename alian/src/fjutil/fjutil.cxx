// fjutil.cxx
#include "fjutil.hh"
#include <iostream>
#include <numeric>
namespace alian
{
	void process_numpy_array(PyObject *array)
	{
		PyArrayObject *np_array = reinterpret_cast<PyArrayObject *>(array);
		if (PyArray_NDIM(np_array) != 1)
		{
			PyErr_SetString(PyExc_TypeError, "Array must be one-dimensional");
			return;
		}

		double *data = static_cast<double *>(PyArray_DATA(np_array));
		npy_intp size = PyArray_SIZE(np_array);

		for (npy_intp i = 0; i < size; ++i)
		{
			std::cout << "Element " << i << ": " << data[i] << std::endl;
		}
	}

	// function to transform three numpy arrays into a vector of PseudoJets
	std::vector<fastjet::PseudoJet> numpy_pxpypz_to_pseudojets(PyObject *px, PyObject *py, PyObject *pz, double m, int index_offset)
	{
		PyArrayObject *np_px = reinterpret_cast<PyArrayObject *>(px);
		PyArrayObject *np_py = reinterpret_cast<PyArrayObject *>(py);
		PyArrayObject *np_pz = reinterpret_cast<PyArrayObject *>(pz);

		if (PyArray_NDIM(np_px) != 1 || PyArray_NDIM(np_py) != 1 || PyArray_NDIM(np_pz) != 1)
		{
			PyErr_SetString(PyExc_TypeError, "Arrays must be one-dimensional");
			return std::vector<fastjet::PseudoJet>();
		}

		npy_intp size = PyArray_SIZE(np_px);
		if (size != PyArray_SIZE(np_py) || size != PyArray_SIZE(np_pz))
		{
			PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
			return std::vector<fastjet::PseudoJet>();
		}

		float *px_data = static_cast<float *>(PyArray_DATA(np_px));
		float *py_data = static_cast<float *>(PyArray_DATA(np_py));
		float *pz_data = static_cast<float *>(PyArray_DATA(np_pz));

		std::vector<fastjet::PseudoJet> particles;
		particles.reserve(size);

		for (npy_intp i = 0; i < size; ++i)
		{
			double E = sqrt(px_data[i] * px_data[i] + py_data[i] * py_data[i] + pz_data[i] * pz_data[i] + m * m);
			particles.emplace_back(px_data[i], py_data[i], pz_data[i], E);
			particles.back().set_user_index(i + index_offset);
		}

		return particles;
	}

	// function to transform three numpy arrays pt,eta,phi into a vector of PseudoJets
	std::vector<fastjet::PseudoJet> numpy_ptetaphi_to_pseudojets(PyObject *pt, PyObject *eta, PyObject *phi, double m, int index_offset)
	{
		PyArrayObject *np_pt = reinterpret_cast<PyArrayObject *>(pt);
		PyArrayObject *np_eta = reinterpret_cast<PyArrayObject *>(eta);
		PyArrayObject *np_phi = reinterpret_cast<PyArrayObject *>(phi);

		if (PyArray_NDIM(np_pt) != 1 || PyArray_NDIM(np_eta) != 1 || PyArray_NDIM(np_phi) != 1)
		{
			PyErr_SetString(PyExc_TypeError, "Arrays must be one-dimensional");
			return std::vector<fastjet::PseudoJet>();
		}

		npy_intp size = PyArray_SIZE(np_pt);
		if (size != PyArray_SIZE(np_eta) || size != PyArray_SIZE(np_phi))
		{
			PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
			return std::vector<fastjet::PseudoJet>();
		}

		float *pt_data  = static_cast<float *>(PyArray_DATA(np_pt));
		float *eta_data = static_cast<float *>(PyArray_DATA(np_eta));
		float *phi_data = static_cast<float *>(PyArray_DATA(np_phi));

		std::vector<fastjet::PseudoJet> particles;
		particles.reserve(size);

		for (npy_intp i = 0; i < size; ++i)
		{
			if (pt_data[i] <= 0.001) // this is 1 MeV (!)
			{
				// PyErr_SetString(PyExc_ValueError, "pt must be positive");
				// return std::vector<fastjet::PseudoJet>();
				continue;
			}
			double px = pt_data[i] * cos(phi_data[i]);
			double py = pt_data[i] * sin(phi_data[i]);
			double pz = pt_data[i] * sinh(eta_data[i]);
			double E = sqrt(px * px + py * py + pz * pz + m * m);

			particles.emplace_back(px, py, pz, E);
			particles.back().set_user_index(i + index_offset);
		}
		return particles;
	}


	// function to transform numpy arrays (px, py, pz, tracksel) into a vector of PseudoJets with track information
	std::vector<fastjet::PseudoJet> numpy_pxpypz_to_tracks(PyObject *px, PyObject *py, PyObject *pz, PyObject *tracksel, int index_offset)
	{
		PyArrayObject *np_px = reinterpret_cast<PyArrayObject *>(px);
		PyArrayObject *np_py = reinterpret_cast<PyArrayObject *>(py);
		PyArrayObject *np_pz = reinterpret_cast<PyArrayObject *>(pz);
		PyArrayObject *np_tracksel = reinterpret_cast<PyArrayObject *>(tracksel);

		if (PyArray_NDIM(np_px) != 1 || PyArray_NDIM(np_py) != 1 || PyArray_NDIM(np_pz) != 1 || PyArray_NDIM(np_tracksel) != 1)
		{
			PyErr_SetString(PyExc_TypeError, "Arrays must be one-dimensional");
			return std::vector<fastjet::PseudoJet>();
		}

		npy_intp size = PyArray_SIZE(np_px);
		if (size != PyArray_SIZE(np_py) || size != PyArray_SIZE(np_pz) || size != PyArray_SIZE(np_tracksel))
		{
			PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
			return std::vector<fastjet::PseudoJet>();
		}

		float *px_data = static_cast<float *>(PyArray_DATA(np_px));
		float *py_data = static_cast<float *>(PyArray_DATA(np_py));
		float *pz_data = static_cast<float *>(PyArray_DATA(np_pz));
		uint8_t *tracksel_data = static_cast<uint8_t *>(PyArray_DATA(np_tracksel));

		std::vector<fastjet::PseudoJet> tracks;
		tracks.reserve(size);

		for (npy_intp i = 0; i < size; ++i)
		{
			// assume the charged pion mass
			double E = sqrt(px_data[i] * px_data[i] + py_data[i] * py_data[i] + pz_data[i] * pz_data[i] + PION_MASS * PION_MASS);
			// charge is stored in the 0th bit of tracksel (1 = positive, 0 = negativ)
			short q = tracksel_data[i] & 0b1 ? 1 : -1;

			tracks.emplace_back(px_data[i], py_data[i], pz_data[i], E);
			tracks.back().set_user_info(new alian::TrackInfo(q, tracksel_data[i]));
			tracks.back().set_user_index(i + index_offset);
		}

		return tracks;
	}

	// function to transform numpy arrays (pT, eta, phi, tracksel) into a vector of PseudoJets with track information
	std::vector<fastjet::PseudoJet> numpy_ptetaphi_to_tracks(PyObject *pt, PyObject *eta, PyObject *phi, PyObject *tracksel, int index_offset)
	{
		PyArrayObject *np_pt = reinterpret_cast<PyArrayObject *>(pt);
		PyArrayObject *np_eta = reinterpret_cast<PyArrayObject *>(eta);
		PyArrayObject *np_phi = reinterpret_cast<PyArrayObject *>(phi);
		PyArrayObject *np_tracksel = reinterpret_cast<PyArrayObject *>(tracksel);

		if (PyArray_NDIM(np_pt) != 1 || PyArray_NDIM(np_eta) != 1 || PyArray_NDIM(np_phi) != 1 || PyArray_NDIM(np_tracksel) != 1)
		{
			PyErr_SetString(PyExc_TypeError, "Arrays must be one-dimensional");
			return std::vector<fastjet::PseudoJet>();
		}

		npy_intp size = PyArray_SIZE(np_pt);
		if (size != PyArray_SIZE(np_eta) || size != PyArray_SIZE(np_phi) || size  != PyArray_SIZE(np_tracksel))
		{
			PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
			return std::vector<fastjet::PseudoJet>();
		}

		float *pt_data  = static_cast<float *>(PyArray_DATA(np_pt));
		float *eta_data = static_cast<float *>(PyArray_DATA(np_eta));
		float *phi_data = static_cast<float *>(PyArray_DATA(np_phi));
		uint8_t *tracksel_data = static_cast<uint8_t *>(PyArray_DATA(np_tracksel));

		std::vector<fastjet::PseudoJet> tracks;
		tracks.reserve(size);

		for (npy_intp i = 0; i < size; ++i)
		{
			if (pt_data[i] <= 0.001) // this is 1 MeV (!)
			{
				// PyErr_SetString(PyExc_ValueError, "pt must be positive");
				return std::vector<fastjet::PseudoJet>();
				continue;
			}
			double px = pt_data[i] * cos(phi_data[i]);
			double py = pt_data[i] * sin(phi_data[i]);
			double pz = pt_data[i] * sinh(eta_data[i]);
			// assume the pion mass
			double E = sqrt(px * px + py * py + pz * pz + PION_MASS * PION_MASS);
			// charge is stored in the 0th bit of tracksel (1 = positive, 0 = negative)
			short q = tracksel_data[i] & 0b1 ? 1 : -1;

			tracks.emplace_back(px, py, pz, E);
			tracks.back().set_user_info(new alian::TrackInfo(q, tracksel_data[i]));
			tracks.back().set_user_index(i + index_offset);
		}

		return tracks;
	}

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
	)
	{
		PyArrayObject *np_energy = reinterpret_cast<PyArrayObject *>(energy);
		PyArrayObject *np_eta = reinterpret_cast<PyArrayObject *>(eta);
		PyArrayObject *np_phi = reinterpret_cast<PyArrayObject *>(phi);
		PyArrayObject *np_m02 = reinterpret_cast<PyArrayObject *>(m02);
		PyArrayObject *np_m20 = reinterpret_cast<PyArrayObject *>(m20);
		PyArrayObject *np_ncells = reinterpret_cast<PyArrayObject *>(ncells);
		PyArrayObject *np_time = reinterpret_cast<PyArrayObject *>(time);
		PyArrayObject *np_exoticity = reinterpret_cast<PyArrayObject *>(exoticity);
		PyArrayObject *np_dbc = reinterpret_cast<PyArrayObject *>(dbc);
		PyArrayObject *np_nlm = reinterpret_cast<PyArrayObject *>(nlm);
		PyArrayObject *np_defn = reinterpret_cast<PyArrayObject *>(defn);
		PyArrayObject *np_matchedTrackN = reinterpret_cast<PyArrayObject *>(matchedTrackN);
		PyArrayObject *np_matchedTrackDeltaEta = reinterpret_cast<PyArrayObject *>(matchedTrackDeltaEta);
		PyArrayObject *np_matchedTrackDeltaPhi = reinterpret_cast<PyArrayObject *>(matchedTrackDeltaPhi);
		PyArrayObject *np_matchedTrackP = reinterpret_cast<PyArrayObject *>(matchedTrackP);
		PyArrayObject *np_matchedTrackPt = reinterpret_cast<PyArrayObject *>(matchedTrackPt);
		PyArrayObject *np_matchedTrackSel = reinterpret_cast<PyArrayObject *>(matchedTrackSel);

		if (PyArray_NDIM(np_energy) != 1
		|| PyArray_NDIM(np_eta) != 1
		|| PyArray_NDIM(np_phi) != 1
		|| PyArray_NDIM(np_m02) != 1
		|| PyArray_NDIM(np_m02) != 1
		|| PyArray_NDIM(np_m20) != 1
		|| PyArray_NDIM(np_ncells) != 1
		|| PyArray_NDIM(np_time) != 1
		|| PyArray_NDIM(np_exoticity) != 1
		|| PyArray_NDIM(np_dbc) != 1
		|| PyArray_NDIM(np_nlm) != 1
		|| PyArray_NDIM(np_defn) != 1
		|| PyArray_NDIM(np_matchedTrackN) != 1
		|| PyArray_NDIM(np_matchedTrackDeltaEta) != 1
		|| PyArray_NDIM(np_matchedTrackDeltaPhi) != 1
		|| PyArray_NDIM(np_matchedTrackP) != 1
		|| PyArray_NDIM(np_matchedTrackPt) != 1
		|| PyArray_NDIM(np_matchedTrackSel) != 1
			)
		{
			PyErr_SetString(PyExc_TypeError, "Arrays must be one-dimensional");
			return std::vector<alian::Cluster>();
		}

		npy_intp size = PyArray_SIZE(np_energy);
		if (size != PyArray_SIZE(np_eta)
			|| size != PyArray_SIZE(np_phi)
			|| size != PyArray_SIZE(np_m02)
			|| size != PyArray_SIZE(np_m02)
			|| size != PyArray_SIZE(np_m20)
			|| size != PyArray_SIZE(np_ncells)
			|| size != PyArray_SIZE(np_time)
			|| size != PyArray_SIZE(np_exoticity)
			|| size != PyArray_SIZE(np_dbc)
			|| size != PyArray_SIZE(np_nlm)
			|| size != PyArray_SIZE(np_defn)
			|| size != PyArray_SIZE(np_matchedTrackN)
		)
		{
			PyErr_SetString(PyExc_ValueError, "Arrays must have the same size");
			return std::vector<alian::Cluster>();
		}

		// if no clusters in the event, return empty vector
		if (size == 0) {
			return std::vector<alian::Cluster>();
		}

		// Data types here have to match those from the NumPy arrays
		// These are taken from the types that uproot reads from the BerkeleyTrees
		float *energy_data   = static_cast<float*> (PyArray_DATA(np_energy));
		float *eta_data      = static_cast<float*> (PyArray_DATA(np_eta));
		float *phi_data      = static_cast<float*> (PyArray_DATA(np_phi));
		float *m02_data      = static_cast<float*> (PyArray_DATA(np_m02));
		float *m20_data      = static_cast<float*> (PyArray_DATA(np_m20));
		int *ncells_data     = static_cast<int*>   (PyArray_DATA(np_ncells));
		float *time_data     = static_cast<float*> (PyArray_DATA(np_time));
		bool *exoticity_data = static_cast<bool*>  (PyArray_DATA(np_exoticity));
		float *dbc_data      = static_cast<float*> (PyArray_DATA(np_dbc));
		int *nlm_data        = static_cast<int*>   (PyArray_DATA(np_nlm));
		int *defn_data       = static_cast<int*>   (PyArray_DATA(np_defn));
		int *matchedTrackN_data  = static_cast<int*> (PyArray_DATA(np_matchedTrackN));
		// float *matchedTrackEta_data  = static_cast<float*> (PyArray_DATA(np_matchedTrackEta));
		// float *matchedTrackPhi_data  = static_cast<float*> (PyArray_DATA(np_matchedTrackPhi));
		// float *matchedTrackP_data  = static_cast<float*> (PyArray_DATA(np_matchedTrackP));

		npy_intp sizeMatchedDeltaEta = PyArray_SIZE(np_matchedTrackDeltaEta);
		npy_intp sizeMatchedDeltaPhi = PyArray_SIZE(np_matchedTrackDeltaPhi);
		npy_intp sizeMatchedP = PyArray_SIZE(np_matchedTrackP);
		npy_intp sizeMatchedPt = PyArray_SIZE(np_matchedTrackPt);
		npy_intp sizeMatchedSel = PyArray_SIZE(np_matchedTrackSel);

		uint16_t nMatchedTracks;
		for (npy_intp i = 0; i < size; ++i) {
			nMatchedTracks += matchedTrackN_data[i];
		}

		if ( nMatchedTracks != sizeMatchedDeltaEta
			|| nMatchedTracks != sizeMatchedDeltaPhi
			|| nMatchedTracks != sizeMatchedP
			|| nMatchedTracks != sizeMatchedPt
			|| nMatchedTracks != sizeMatchedSel)
		{
			PyErr_SetString(PyExc_ValueError, "Number of matched tracks does not fit");
			return std::vector<alian::Cluster>();
		}

		std::vector<alian::Cluster> clusters;
		clusters.reserve(size);

		int iMatchedTrack(0);
		for (npy_intp i = 0; i < size; ++i)
		{
			// assume massless in this calculation
			double pt = energy_data[i] / cosh(eta_data[i]);
			// double pz = energy_data[i] * tanh(eta_data[i]);
			// if (pt <= 0.001) // this is 1 MeV (!)
			// {
			// 	// PyErr_SetString(PyExc_ValueError, "pt must be positive");
			// 	return std::vector<alian::Cluster>();
			// 	continue;
			// }
			double px = pt * cos(phi_data[i]);
			double py = pt * sin(phi_data[i]);
			double pz = pt * sinh(eta_data[i]);

			npy_intp const dims[1] = {matchedTrackN_data[i]};
			PyArrayObject* matchedTrackDeltaEta = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_FLOAT));
			PyArrayObject* matchedTrackDeltaPhi = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_FLOAT));
			PyArrayObject* matchedTrackP = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_FLOAT));
			PyArrayObject* matchedTrackPt = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_FLOAT));
			PyArrayObject* matchedTrackSel = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(1, dims, NPY_UINT8));
			
			for (int iTrack = 0; iTrack < matchedTrackN_data[i]; ++iTrack) {
				*(npy_float *)PyArray_GETPTR1(matchedTrackDeltaEta, iTrack) = *(npy_float *)PyArray_GETPTR1(np_matchedTrackDeltaEta, iMatchedTrack);
				*(npy_float *)PyArray_GETPTR1(matchedTrackDeltaPhi, iTrack) = *(npy_float *)PyArray_GETPTR1(np_matchedTrackDeltaPhi, iMatchedTrack);
				*(npy_float *)PyArray_GETPTR1(matchedTrackP, iTrack) = *(npy_float *)PyArray_GETPTR1(np_matchedTrackP, iMatchedTrack);
				*(npy_float *)PyArray_GETPTR1(matchedTrackPt, iTrack) = *(npy_float *)PyArray_GETPTR1(np_matchedTrackPt, iMatchedTrack);
				*(npy_uint8 *)PyArray_GETPTR1(matchedTrackSel, iTrack) = *(npy_uint8 *)PyArray_GETPTR1(np_matchedTrackSel, iMatchedTrack);
				iMatchedTrack++;
			}
			clusters.emplace_back(px, py, pz,
				energy_data[i],
				m02_data[i],
				m20_data[i],
				ncells_data[i],
				time_data[i],
				exoticity_data[i],
				dbc_data[i],
				nlm_data[i],
				defn_data[i],
				matchedTrackN_data[i],
				(PyObject*)matchedTrackDeltaEta,
				(PyObject*)matchedTrackDeltaPhi,
				(PyObject*)matchedTrackP,
				(PyObject*)matchedTrackPt,
				(PyObject*)matchedTrackSel
			);
			clusters.back().set_user_index(i + index_offset);
		}
		if (iMatchedTrack != nMatchedTracks) {
			PyErr_SetString(PyExc_ValueError, "Somehow did not consume all matched tracks");
			return std::vector<alian::Cluster>();
		}
		return clusters;
	}
}