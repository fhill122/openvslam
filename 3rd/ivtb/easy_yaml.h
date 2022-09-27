/*
* Created by Ivan B on 2022/7/25.
*/

#ifndef IVTB_PARAM_EASY_YAML_H_
#define IVTB_PARAM_EASY_YAML_H_

#include <yaml-cpp/yaml.h>

#include <iostream>

namespace ivtb{

inline std::string GetYamlType(const YAML::Node &node){
   switch (node.Type()) {
       case YAML::NodeType::Null: return "Null";
       case YAML::NodeType::Scalar: return "Scalar";
       case YAML::NodeType::Sequence: return "Sequence";
       case YAML::NodeType::Map: return "Map";
       case YAML::NodeType::Undefined: return "Undefined";
       default:
           // yaml-cpp api has changed
           abort();
   }
}

template<typename T>
void OptionalSet(const YAML::Node &node, T &out, bool warning=false,
                const char *err_key = ""){
   try{
       if (node)
           out = node.template as<T>();
       else if (warning)
           fprintf(stderr, "%s not found\n", err_key);
   } catch (const std::exception &exc){
       if (warning)
           fprintf(stderr, "failed to set %s: %s\n",
                   err_key, exc.what());
   }
}

inline YAML::Node YamlNodeFromDotPath(const YAML::Node &node, const std::vector<std::string> &path){
   if (path.empty()) {
       return {};
   } else if (path.size()==1){
       return node[path[0]];
   }

   // option 1: clone
   // note: yaml-cpp is buggy, Clone here is necessary, otherwise node would be changed
   // YAML::Node out = YAML::Clone(node);
   // for (const auto &str : path) {
   //     out = out[str];
   // }
   // return out;

   // option 2: avoid clone, save all intermedia nodes
   std::vector<YAML::Node> nodes(path.size()-1);
   for (int i = 0; i < path.size(); ++i) {
       if (i==0){
           nodes[i] = node[path[i]];
       }
       else if (i==path.size()-1){
           // operator [] returns a node not a reference
           return nodes[i-1][path[i]];
       }
       else {
           nodes[i] = nodes[i-1][path[i]];
       }
   }
   return {}; // would never reach here;
}

inline std::vector<std::string> SplitDotPath(const std::string &path){
   constexpr char kDelim = '.';
   std::vector<std::string> out;
   size_t start;
   size_t end = 0;
   while ((start = path.find_first_not_of(kDelim, end)) != std::string::npos) {
       end = path.find(kDelim, start);
       out.push_back(path.substr(start, end - start));
   }
   return out;
}

}

#define SET_FROM_YAML(conf, var) \
   try{                         \
       ivtb::OptionalSet(ivtb::YamlNodeFromDotPath(conf, ivtb::SplitDotPath(#var)), var, true, #var); \
   } catch (const std::exception &exc){               \
       fprintf(stderr, "failed to set %s: %s\n", #var, exc.what()); \
   }

// set silently, ignore exception without any warning
#define SET_FROM_YAML_S(conf, var) \
   try{                         \
       ivtb::OptionalSet(ivtb::YamlNodeFromDotPath(conf, ivtb::SplitDotPath(#var)),var); \
   } catch (...){}

// todo ivan. consider set with variadic macro, SET_FROM_YAML(conf, ...)
//  maybe could be implemented like: https://stackoverflow.com/a/45586169/12392479

#endif //IVTB_PARAM_EASY_YAML_H_
