#ifndef AI_MEMORY_FFI_H
#define AI_MEMORY_FFI_H

#include <stdint.h>

char *ai_memory_ingest_paths(const char *store_path, const char *paths_json);
char *ai_memory_rebuild_memory(const char *store_path);
char *ai_memory_get_memory_summary(const char *store_path);
char *ai_memory_get_visualization_snapshot(const char *store_path);
char *ai_memory_run_query(const char *store_path, const char *query_json);
char *ai_memory_grep_region(const char *store_path, const char *region_id, const char *needle);
char *ai_memory_open_document_excerpt(const char *store_path, const char *chunk_id);
char *ai_memory_grep_document(const char *store_path, const char *document_id, const char *needle);
char *ai_memory_semantic_document_search(const char *store_path, const char *document_id, const char *query);
char *ai_memory_surf_open(const char *store_path, const char *node_json);
char *ai_memory_surf_neighbors(const char *store_path, const char *node_json, uintptr_t max_results);
char *ai_memory_surf_expand(const char *store_path, const char *chunk_id, const char *mode_json, uintptr_t window);
char *ai_memory_surf_jump_to_anchor(const char *store_path, const char *anchor_id, uintptr_t window);
char *ai_memory_surf_session_step(const char *store_path, const char *session_json, const char *action_json);
char *ai_memory_list_links(const char *store_path, const char *node_json);
char *ai_memory_step_navigation(const char *store_path, const char *session_json, const char *link_id);
char *ai_memory_backtrack_navigation(const char *store_path, const char *session_json);
char *ai_memory_save_model_config(const char *store_path, const char *config_json);
char *ai_memory_load_model_config(const char *store_path);
char *ai_memory_test_model_connection(const char *config_json);
void ai_memory_free_string(char *ptr);

#endif
