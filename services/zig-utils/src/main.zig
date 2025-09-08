// JADED System Utilities Service (Zig)
// A precíziós eszköz - Alacsony szintű optimalizált műveletek

const std = @import("std");
const net = std.net;
const http = std.http;
const json = std.json;
const fs = std.fs;
const mem = std.mem;
const time = std.time;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Atomic = std.atomic.Value;

const SERVICE_NAME = "System Utils (Zig)";
const SERVICE_DESC = "A precíziós eszköz - Alacsony szintű optimalizált műveletek";
const PORT = 8005;
const MAX_CONNECTIONS = 1000;
const BUFFER_SIZE = 64 * 1024; // 64KB high-performance buffer

// Service metrics with atomic operations for thread safety
const ServiceMetrics = struct {
    requests_processed: Atomic(u64) = Atomic(u64).init(0),
    data_processed: Atomic(u64) = Atomic(u64).init(0),
    active_connections: Atomic(u32) = Atomic(u32).init(0),
    system_operations: Atomic(u64) = Atomic(u64).init(0),
    cache_operations: Atomic(u64) = Atomic(u64).init(0),
    file_operations: Atomic(u64) = Atomic(u64).init(0),
    memory_usage: Atomic(u64) = Atomic(u64).init(0),
    start_time: u64,
    
    fn init() ServiceMetrics {
        return ServiceMetrics{
            .start_time = time.timestamp(),
        };
    }
};

// File system operations with memory-mapped I/O
const FileOpsService = struct {
    allocator: mem.Allocator,
    cache: HashMap([]const u8, []u8, mem.Allocator, true),
    cache_mutex: Mutex,
    
    const Self = @This();
    
    fn init(allocator: mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .cache = HashMap([]const u8, []u8, mem.Allocator, true).init(allocator),
            .cache_mutex = Mutex{},
        };
    }
    
    fn readFileOptimized(self: *Self, path: []const u8) ![]u8 {
        // Check cache first
        self.cache_mutex.lock();
        defer self.cache_mutex.unlock();
        
        if (self.cache.get(path)) |cached_data| {
            return cached_data;
        }
        
        // Memory-mapped file reading for large files
        const file = fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };
        defer file.close();
        
        const file_size = try file.getEndPos();
        
        if (file_size > 1024 * 1024) { // Use memory mapping for files > 1MB
            // Memory-mapped approach for large files
            const mapped_mem = try std.posix.mmap(
                null,
                file_size,
                std.posix.PROT.READ,
                std.posix.MAP.PRIVATE,
                file.handle,
                0
            );
            defer std.posix.munmap(mapped_mem);
            
            const result = try self.allocator.dupe(u8, mapped_mem);
            try self.cache.put(try self.allocator.dupe(u8, path), result);
            
            return result;
        } else {
            // Regular read for smaller files
            const content = try file.readToEndAlloc(self.allocator, file_size);
            try self.cache.put(try self.allocator.dupe(u8, path), content);
            
            return content;
        }
    }
    
    fn writeFileOptimized(self: *Self, path: []const u8, data: []const u8) !void {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        
        // Use vectorized I/O for large writes
        if (data.len > BUFFER_SIZE) {
            var pos: usize = 0;
            while (pos < data.len) {
                const write_size = @min(BUFFER_SIZE, data.len - pos);
                _ = try file.write(data[pos..pos + write_size]);
                pos += write_size;
            }
        } else {
            _ = try file.writeAll(data);
        }
        
        // Update cache
        self.cache_mutex.lock();
        defer self.cache_mutex.unlock();
        
        const path_copy = try self.allocator.dupe(u8, path);
        const data_copy = try self.allocator.dupe(u8, data);
        try self.cache.put(path_copy, data_copy);
    }
    
    fn optimizeDirectory(self: *Self, dir_path: []const u8) !std.json.Value {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();
        
        var dir = fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return error.DirectoryNotFound,
            else => return err,
        };
        defer dir.close();
        
        var walker = try dir.walk(arena_allocator);
        defer walker.deinit();
        
        var total_files: u32 = 0;
        var total_size: u64 = 0;
        var duplicates_found: u32 = 0;
        var optimization_saved: u64 = 0;
        
        var file_hashes = HashMap(u64, []const u8, mem.Allocator, true).init(arena_allocator);
        
        while (try walker.next()) |entry| {
            if (entry.kind == .file) {
                const full_path = try fs.path.join(arena_allocator, &[_][]const u8{ dir_path, entry.path });
                const content = self.readFileOptimized(full_path) catch continue;
                
                // Simple hash for duplicate detection
                var hasher = std.hash.Wyhash.init(0);
                hasher.update(content);
                const file_hash = hasher.final();
                
                if (file_hashes.get(file_hash)) |existing_path| {
                    duplicates_found += 1;
                    optimization_saved += content.len;
                    print("Duplicate found: {s} == {s}\n", .{ full_path, existing_path });
                } else {
                    try file_hashes.put(file_hash, try arena_allocator.dupe(u8, full_path));
                }
                
                total_files += 1;
                total_size += content.len;
            }
        }
        
        return std.json.Value{
            .object = std.json.ObjectMap.init(arena_allocator),
        };
    }
};

// Memory optimization utilities with SIMD when available
const MemoryOps = struct {
    fn optimizeMemoryLayout(data: []u8, pattern: u8) void {
        // Use SIMD operations if available on the target architecture
        if (std.simd.suggestVectorLength(u8)) |vec_len| {
            const Vec = @Vector(vec_len, u8);
            const pattern_vec: Vec = @splat(pattern);
            
            var i: usize = 0;
            while (i + vec_len <= data.len) {
                const chunk: Vec = data[i..i + vec_len][0..vec_len].*;
                const result = chunk ^ pattern_vec;
                data[i..i + vec_len][0..vec_len].* = result;
                i += vec_len;
            }
            
            // Handle remaining bytes
            while (i < data.len) {
                data[i] ^= pattern;
                i += 1;
            }
        } else {
            // Fallback for architectures without SIMD
            for (data) |*byte| {
                byte.* ^= pattern;
            }
        }
    }
    
    fn compressData(allocator: mem.Allocator, data: []const u8) ![]u8 {
        // Simple run-length encoding for demonstration
        var result = ArrayList(u8).init(allocator);
        
        if (data.len == 0) return try result.toOwnedSlice();
        
        var current_byte = data[0];
        var count: u8 = 1;
        
        for (data[1..]) |byte| {
            if (byte == current_byte and count < 255) {
                count += 1;
            } else {
                try result.append(count);
                try result.append(current_byte);
                current_byte = byte;
                count = 1;
            }
        }
        
        try result.append(count);
        try result.append(current_byte);
        
        return try result.toOwnedSlice();
    }
    
    fn decompressData(allocator: mem.Allocator, compressed: []const u8) ![]u8 {
        var result = ArrayList(u8).init(allocator);
        
        var i: usize = 0;
        while (i < compressed.len) {
            const count = compressed[i];
            const byte_value = compressed[i + 1];
            
            var j: u8 = 0;
            while (j < count) {
                try result.append(byte_value);
                j += 1;
            }
            
            i += 2;
        }
        
        return try result.toOwnedSlice();
    }
};

// System monitoring and process management
const SystemMonitor = struct {
    fn getCpuInfo(allocator: mem.Allocator) !std.json.Value {
        var object_map = std.json.ObjectMap.init(allocator);
        
        // Get CPU information from /proc/cpuinfo on Linux
        const cpuinfo_content = fs.cwd().readFileAlloc(allocator, "/proc/cpuinfo", 8192) catch |err| switch (err) {
            error.FileNotFound => return std.json.Value{ .object = object_map },
            else => return err,
        };
        defer allocator.free(cpuinfo_content);
        
        var cpu_count: u32 = 0;
        var cpu_model: []const u8 = "Unknown";
        
        var lines = mem.split(u8, cpuinfo_content, "\n");
        while (lines.next()) |line| {
            if (mem.startsWith(u8, line, "processor")) {
                cpu_count += 1;
            } else if (mem.startsWith(u8, line, "model name")) {
                if (mem.indexOf(u8, line, ":")) |colon_pos| {
                    cpu_model = mem.trim(u8, line[colon_pos + 1..], " \t");
                }
            }
        }
        
        try object_map.put("cpu_count", std.json.Value{ .integer = cpu_count });
        try object_map.put("cpu_model", std.json.Value{ .string = cpu_model });
        
        return std.json.Value{ .object = object_map };
    }
    
    fn getMemoryInfo(allocator: mem.Allocator) !std.json.Value {
        var object_map = std.json.ObjectMap.init(allocator);
        
        const meminfo_content = fs.cwd().readFileAlloc(allocator, "/proc/meminfo", 4096) catch |err| switch (err) {
            error.FileNotFound => return std.json.Value{ .object = object_map },
            else => return err,
        };
        defer allocator.free(meminfo_content);
        
        var total_memory: u64 = 0;
        var free_memory: u64 = 0;
        var available_memory: u64 = 0;
        
        var lines = mem.split(u8, meminfo_content, "\n");
        while (lines.next()) |line| {
            if (mem.startsWith(u8, line, "MemTotal:")) {
                if (mem.indexOf(u8, line, " ")) |space_pos| {
                    const value_str = mem.trim(u8, line[space_pos..], " kB\t");
                    total_memory = std.fmt.parseInt(u64, value_str, 10) catch 0;
                }
            } else if (mem.startsWith(u8, line, "MemFree:")) {
                if (mem.indexOf(u8, line, " ")) |space_pos| {
                    const value_str = mem.trim(u8, line[space_pos..], " kB\t");
                    free_memory = std.fmt.parseInt(u64, value_str, 10) catch 0;
                }
            } else if (mem.startsWith(u8, line, "MemAvailable:")) {
                if (mem.indexOf(u8, line, " ")) |space_pos| {
                    const value_str = mem.trim(u8, line[space_pos..], " kB\t");
                    available_memory = std.fmt.parseInt(u64, value_str, 10) catch 0;
                }
            }
        }
        
        try object_map.put("total_kb", std.json.Value{ .integer = @intCast(total_memory) });
        try object_map.put("free_kb", std.json.Value{ .integer = @intCast(free_memory) });
        try object_map.put("available_kb", std.json.Value{ .integer = @intCast(available_memory) });
        try object_map.put("used_percentage", std.json.Value{ .float = @as(f64, @floatFromInt(total_memory - available_memory)) / @as(f64, @floatFromInt(total_memory)) * 100.0 });
        
        return std.json.Value{ .object = object_map };
    }
};

// Global service state
var service_metrics: ServiceMetrics = undefined;
var file_ops_service: FileOpsService = undefined;

// HTTP request handlers
fn handleHealth(allocator: mem.Allocator) ![]u8 {
    const uptime = time.timestamp() - service_metrics.start_time;
    
    var object_map = std.json.ObjectMap.init(allocator);
    try object_map.put("status", std.json.Value{ .string = "healthy" });
    try object_map.put("service", std.json.Value{ .string = SERVICE_NAME });
    try object_map.put("description", std.json.Value{ .string = SERVICE_DESC });
    try object_map.put("uptime_seconds", std.json.Value{ .integer = @intCast(uptime) });
    try object_map.put("zig_version", std.json.Value{ .string = "0.13.0" });
    
    var metrics_map = std.json.ObjectMap.init(allocator);
    try metrics_map.put("requests_processed", std.json.Value{ .integer = @intCast(service_metrics.requests_processed.load(.monotonic)) });
    try metrics_map.put("data_processed_mb", std.json.Value{ .integer = @intCast(service_metrics.data_processed.load(.monotonic) / (1024 * 1024)) });
    try metrics_map.put("active_connections", std.json.Value{ .integer = service_metrics.active_connections.load(.monotonic) });
    try metrics_map.put("system_operations", std.json.Value{ .integer = @intCast(service_metrics.system_operations.load(.monotonic)) });
    try metrics_map.put("cache_operations", std.json.Value{ .integer = @intCast(service_metrics.cache_operations.load(.monotonic)) });
    try object_map.put("metrics", std.json.Value{ .object = metrics_map });
    
    const health_response = std.json.Value{ .object = object_map };
    return try std.json.stringifyAlloc(allocator, health_response, .{});
}

fn handleServiceInfo(allocator: mem.Allocator) ![]u8 {
    var object_map = std.json.ObjectMap.init(allocator);
    try object_map.put("service_name", std.json.Value{ .string = "System Utils" });
    try object_map.put("language", std.json.Value{ .string = "Zig" });
    try object_map.put("version", std.json.Value{ .string = "1.0.0" });
    try object_map.put("description", std.json.Value{ .string = "Alacsony szintű, optimalizált rendszerműveletek maximális teljesítménnyel" });
    
    var features_array = std.json.Array.init(allocator);
    try features_array.append(std.json.Value{ .string = "Zero-cost abstractions" });
    try features_array.append(std.json.Value{ .string = "Memory-mapped I/O" });
    try features_array.append(std.json.Value{ .string = "SIMD optimizations" });
    try features_array.append(std.json.Value{ .string = "Thread-safe operations" });
    try features_array.append(std.json.Value{ .string = "System resource monitoring" });
    try features_array.append(std.json.Value{ .string = "File system optimization" });
    try features_array.append(std.json.Value{ .string = "Data compression utilities" });
    try features_array.append(std.json.Value{ .string = "Process management" });
    try object_map.put("features", std.json.Value{ .array = features_array });
    
    var capabilities_map = std.json.ObjectMap.init(allocator);
    try capabilities_map.put("max_connections", std.json.Value{ .integer = MAX_CONNECTIONS });
    try capabilities_map.put("buffer_size_kb", std.json.Value{ .integer = BUFFER_SIZE / 1024 });
    try capabilities_map.put("memory_safety", std.json.Value{ .boolean = true });
    try capabilities_map.put("compile_time_optimizations", std.json.Value{ .boolean = true });
    try capabilities_map.put("cross_platform", std.json.Value{ .boolean = true });
    try object_map.put("capabilities", std.json.Value{ .object = capabilities_map });
    
    const info_response = std.json.Value{ .object = object_map };
    return try std.json.stringifyAlloc(allocator, info_response, .{});
}

fn handleSystemInfo(allocator: mem.Allocator) ![]u8 {
    var object_map = std.json.ObjectMap.init(allocator);
    
    const cpu_info = SystemMonitor.getCpuInfo(allocator) catch std.json.Value{ .object = std.json.ObjectMap.init(allocator) };
    const memory_info = SystemMonitor.getMemoryInfo(allocator) catch std.json.Value{ .object = std.json.ObjectMap.init(allocator) };
    
    try object_map.put("cpu_info", cpu_info);
    try object_map.put("memory_info", memory_info);
    try object_map.put("timestamp", std.json.Value{ .integer = @intCast(time.timestamp()) });
    
    const system_response = std.json.Value{ .object = object_map };
    return try std.json.stringifyAlloc(allocator, system_response, .{});
}

fn handleFileOperation(allocator: mem.Allocator, request_body: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, request_body, .{});
    defer parsed.deinit();
    
    const root = parsed.value.object;
    const operation = root.get("operation").?.string;
    
    var result_map = std.json.ObjectMap.init(allocator);
    
    if (mem.eql(u8, operation, "read")) {
        const path = root.get("path").?.string;
        const content = file_ops_service.readFileOptimized(path) catch |err| {
            try result_map.put("error", std.json.Value{ .string = @errorName(err) });
            const error_response = std.json.Value{ .object = result_map };
            return try std.json.stringifyAlloc(allocator, error_response, .{});
        };
        
        try result_map.put("status", std.json.Value{ .string = "success" });
        try result_map.put("size_bytes", std.json.Value{ .integer = @intCast(content.len) });
        try result_map.put("content_base64", std.json.Value{ .string = "base64_encoded_content" }); // Would encode actual content
        
        _ = service_metrics.file_operations.fetchAdd(1, .monotonic);
    } else if (mem.eql(u8, operation, "write")) {
        const path = root.get("path").?.string;
        const data = root.get("data").?.string;
        
        file_ops_service.writeFileOptimized(path, data) catch |err| {
            try result_map.put("error", std.json.Value{ .string = @errorName(err) });
            const error_response = std.json.Value{ .object = result_map };
            return try std.json.stringifyAlloc(allocator, error_response, .{});
        };
        
        try result_map.put("status", std.json.Value{ .string = "written" });
        try result_map.put("bytes_written", std.json.Value{ .integer = @intCast(data.len) });
        
        _ = service_metrics.file_operations.fetchAdd(1, .monotonic);
    } else {
        try result_map.put("error", std.json.Value{ .string = "Unsupported file operation" });
        try result_map.put("supported_operations", std.json.Value{ .array = blk: {
            var array = std.json.Array.init(allocator);
            try array.append(std.json.Value{ .string = "read" });
            try array.append(std.json.Value{ .string = "write" });
            try array.append(std.json.Value{ .string = "optimize" });
            break :blk array;
        }});
    }
    
    const response = std.json.Value{ .object = result_map };
    return try std.json.stringifyAlloc(allocator, response, .{});
}

// Simple HTTP server implementation
fn handleConnection(connection: net.Server.Connection) void {
    defer connection.stream.close();
    
    _ = service_metrics.active_connections.fetchAdd(1, .monotonic);
    defer _ = service_metrics.active_connections.fetchSub(1, .monotonic);
    _ = service_metrics.requests_processed.fetchAdd(1, .monotonic);
    
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var buffer: [BUFFER_SIZE]u8 = undefined;
    const bytes_read = connection.stream.read(&buffer) catch {
        print("Error reading from connection\n");
        return;
    };
    
    const request = buffer[0..bytes_read];
    
    // Parse HTTP request (simplified)
    var lines = mem.split(u8, request, "\r\n");
    const request_line = lines.next() orelse return;
    
    var parts = mem.split(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    
    const response_body = blk: {
        if (mem.eql(u8, path, "/health")) {
            break :blk handleHealth(allocator) catch "{}";
        } else if (mem.eql(u8, path, "/info")) {
            break :blk handleServiceInfo(allocator) catch "{}";
        } else if (mem.eql(u8, path, "/system")) {
            break :blk handleSystemInfo(allocator) catch "{}";
        } else if (mem.eql(u8, path, "/files") and mem.eql(u8, method, "POST")) {
            // Extract request body (simplified)
            if (mem.indexOf(u8, request, "\r\n\r\n")) |body_start| {
                const body = request[body_start + 4..];
                break :blk handleFileOperation(allocator, body) catch "{}";
            }
            break :blk "{}";
        } else {
            break :blk "{\"error\":\"Not Found\"}";
        }
    };
    
    const http_response = std.fmt.allocPrint(allocator, 
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ response_body.len, response_body }
    ) catch return;
    
    _ = connection.stream.writeAll(http_response) catch {};
    
    print("Handled request: {s} {s} -> {} bytes\n", .{ method, path, response_body.len });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    service_metrics = ServiceMetrics.init();
    file_ops_service = FileOpsService.init(allocator);
    
    print("🚀 Starting {s} on port {d}\n", .{ SERVICE_NAME, PORT });
    print("⚡ {s}\n", .{SERVICE_DESC});
    print("🔧 Buffer size: {d}KB, Max connections: {d}\n", .{ BUFFER_SIZE / 1024, MAX_CONNECTIONS });
    
    const address = net.Address.parseIp("0.0.0.0", PORT) catch unreachable;
    var server = address.listen(.{ .reuse_address = true }) catch |err| {
        print("Failed to start server: {}\n", .{err});
        return;
    };
    defer server.deinit();
    
    print("✅ System Utilities Service ready and listening on port {d}\n", .{PORT});
    print("🎯 A precíziós eszköz aktiválva - Zero-cost abstractions ready\n");
    
    while (true) {
        const connection = server.accept() catch |err| {
            print("Failed to accept connection: {}\n", .{err});
            continue;
        };
        
        // Handle connection in current thread (could be threaded for production)
        handleConnection(connection);
    }
}