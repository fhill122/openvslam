#ifndef OPENVSLAM_DATA_MAP_DATABASE_H
#define OPENVSLAM_DATA_MAP_DATABASE_H

#include "openvslam/data/bow_vocabulary.h"

#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>

#include <nlohmann/json_fwd.hpp>

namespace openvslam {

namespace camera {
class base;
} // namespace camera

namespace data {

class frame;
class keyframe;
class landmark;
class camera_database;

class map_database {
public:
    /**
     * Constructor
     */
    map_database();

    /**
     * Destructor
     */
    ~map_database();

    /**
     * Add keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Add landmark to the database
     * @param lm
     */
    void add_landmark(std::shared_ptr<landmark>& lm);

    /**
     * Erase landmark from the database
     * @param lm
     */
    void erase_landmark(unsigned int id);

    /**
     * Set local landmarks
     * @param local_lms
     */
    void set_local_landmarks(const std::vector<std::shared_ptr<landmark>>& local_lms);

    /**
     * Get local landmarks
     * @return
     */
    std::vector<std::shared_ptr<landmark>> get_local_landmarks() const;

    std::shared_ptr<keyframe> getKeyframe(std::uint32_t id) const{
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        auto itr = keyframes_.find(id);
        return itr==keyframes_.end()? nullptr : itr->second;
    }

    template< template <class, class...> class ContainerT>
    ContainerT<std::shared_ptr<data::keyframe>> getKeyframes(int id_start, int id_end) const{
        CV_Assert(id_end>id_start);
        id_start = std::max(0, id_start);
        ContainerT<std::shared_ptr<data::keyframe>> out;
        out.reserve(id_end-id_start);
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        for (auto i = id_start; i < id_end; ++i) {
            auto itr = keyframes_.find(i);
            if (itr!=keyframes_.end()) out.insert(out.end(),itr->second);
        }
        return out;
    }

    /**
     * Get all of the keyframes in the database
     * @return
     */
    std::vector<std::shared_ptr<keyframe>> get_all_keyframes() const;

    /**
     * Get the number of keyframes
     * @return
     */
    unsigned get_num_keyframes() const;

    /**
     * Get all of the landmarks in the database
     * @return
     */
    std::vector<std::shared_ptr<landmark>> get_all_landmarks() const;

    /**
     * Get the number of landmarks
     * @return
     */
    unsigned int get_num_landmarks() const;

    /**
     * Clear the database
     */
    void clear();

    //! origin keyframe
    std::shared_ptr<keyframe> origin_keyfrm_ = nullptr;

    //! mutex for locking ALL access to the database
    //! (NOTE: cannot used in map_database class)
    static std::mutex mtx_database_;

private:
    //! mutex for mutual exclusion controll between class methods
    mutable std::mutex mtx_map_access_;

    //-----------------------------------------
    // keyframe and landmark database

    //! IDs and keyframes
    std::unordered_map<unsigned int, std::shared_ptr<keyframe>> keyframes_;
    //! IDs and landmarks
    std::unordered_map<unsigned int, std::shared_ptr<landmark>> landmarks_;

    //! local landmarks
    std::vector<std::shared_ptr<landmark>> local_landmarks_;

    //! max keyframe ID
    unsigned int max_keyfrm_id_ = 0;

};

} // namespace data
} // namespace openvslam

#endif // OPENVSLAM_DATA_MAP_DATABASE_H
