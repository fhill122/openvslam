// #ifndef OPENVSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
// #define OPENVSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
//
// namespace openvslam {
//
// namespace data {
// class map_database;
// } // namespace data
//
// namespace optimize {
//
// class global_bundle_adjuster {
// public:
//     /**
//      * Constructor
//      * @param map_db
//      * @param num_iter
//      * @param use_huber_kernel
//      */
//     explicit global_bundle_adjuster(data::map_database* map_db, const unsigned int num_iter = 10, const bool use_huber_kernel = true);
//
//     /**
//      * Destructor
//      */
//     virtual ~global_bundle_adjuster() = default;
//
//     /**
//      * Perform optimization
//      */
//     void optimize() const;
//
// private:
//     //! map database
//     const data::map_database* map_db_;
//
//     //! number of iterations of optimization
//     unsigned int num_iter_;
//
//     //! use Huber loss or not
//     const bool use_huber_kernel_;
// };
//
// } // namespace optimize
// } // namespace openvslam
//
// #endif // OPENVSLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
