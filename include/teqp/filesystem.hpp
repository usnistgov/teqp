#pragma once

#include <filesystem>

namespace teqp {
    /** Find all files in a given folder that match the specified file extension
     *  The extension should be specified with the period, as in ".json"
     */
    static auto get_files_in_folder(const std::string& folder, const std::string& extension)
    {
        std::vector<std::filesystem::path> files;
        for (auto const& dir_entry : std::filesystem::directory_iterator{ folder }) {
            auto path = dir_entry.path();
            if (path.extension() == extension) {
                files.push_back(path);
            }
        }
        return files;
    }
}; // namespace teqp