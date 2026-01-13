#pragma once

#include <map/defines.h>
#include <map/observation.h>

#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace map {

// Index type for observation storage
using ObservationIndex = size_t;
constexpr ObservationIndex INVALID_OBSERVATION_INDEX =
    static_cast<ObservationIndex>(-1);

class TracksManager {
 public:
  void AddObservation(const ShotId& shot_id, const TrackId& track_id,
                      const Observation& observation);
  Observation GetObservation(const ShotId& shot, const TrackId& track) const;

  int NumShots() const;
  int NumTracks() const;
  std::vector<ShotId> GetShotIds() const;
  std::vector<TrackId> GetTrackIds() const;

  // Returns a map of track_id -> observation for a given shot
  // Note: This constructs the map on each call (no longer returns a reference)
  std::unordered_map<TrackId, Observation> GetShotObservations(
      const ShotId& shot) const;
  // Returns a map of shot_id -> observation for a given track
  // Note: This constructs the map on each call (no longer returns a reference)
  std::unordered_map<ShotId, Observation> GetTrackObservations(
      const TrackId& track) const;

  TracksManager ConstructSubTracksManager(
      const std::vector<TrackId>& tracks,
      const std::vector<ShotId>& shots) const;

  using KeyPointTuple = std::tuple<TrackId, Observation, Observation>;
  std::vector<KeyPointTuple> GetAllCommonObservations(
      const ShotId& shot1, const ShotId& shot2) const;
  std::tuple<std::vector<map::TrackId>, MatX2f, MatX2f>
  GetAllCommonObservationsArrays(const ShotId& shot1,
                                 const ShotId& shot2) const;

  using ShotPair = std::pair<ShotId, ShotId>;
  std::unordered_map<ShotPair, int, HashPair> GetAllPairsConnectivity(
      const std::vector<ShotId>& shots,
      const std::vector<TrackId>& tracks) const;

  static TracksManager InstanciateFromFile(const std::string& filename);
  void WriteToFile(const std::string& filename) const;

  static TracksManager InstanciateFromString(const std::string& str);
  std::string AsString() const;

  static TracksManager MergeTracksManager(
      const std::vector<const TracksManager*>& tracks_manager);

  bool HasShotObservations(const ShotId& shot) const;

  static std::string TRACKS_HEADER;
  static int TRACKS_VERSION;

 private:
  // Interning types and helpers
  using StringId = size_t;
  StringId GetShotIndex(const ShotId& id);
  StringId GetTrackIndex(const TrackId& id);
  StringId GetOrInsertShotIndex(const ShotId& id);
  StringId GetOrInsertTrackIndex(const TrackId& id);

  // Single storage for all observations - each observation stored exactly once
  std::vector<Observation> observations_;

  // Interning storage
  std::vector<ShotId> shot_ids_;
  std::vector<TrackId> track_ids_;
  std::unordered_map<ShotId, StringId> shot_id_to_index_;
  std::unordered_map<TrackId, StringId> track_id_to_index_;

  // Adjacency lists using integer indices
  // tracks_per_shot_[shot_index] -> map {track_index -> obs_index}
  std::vector<std::unordered_map<StringId, ObservationIndex>> tracks_per_shot_;
  // shots_per_track_[track_index] -> map {shot_index -> obs_index}
  std::vector<std::unordered_map<StringId, ObservationIndex>> shots_per_track_;
};
}  // namespace map
