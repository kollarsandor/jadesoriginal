// JADED System Utils Service (Zig)
// Zero-cost absztrakciók - Fájlrendszer optimalizáció és rendszermonitorozás
// Compile-time optimalizációk nagy teljesítményű I/O műveletekhez

const std = @import("std");
const print = std.debug.print;
const Thread = std.Thread;
const Atomic = std.atomic.Atomic;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Allocator = std.mem.Allocator;

const PORT: u16 = 8005;
const MAX_CONNECTIONS: u32 = 1000;
const BUFFER_SIZE: usize = 65536;
const MONITORING_INTERVAL_MS: u64 = 5000;

// Compile-time configuration
const Config = struct {
    max_file_size: comptime_int = 1 * 1024 * 1024 * 1024, // 1GB
    max_concurrent_operations: comptime_int = 100,
    enable_simd: bool = true,
    enable_async_io: bool = true,
    log_level: LogLevel = .info,
};

const LogLevel = enum { debug, info, warn, err };
const config = Config{};

// Zero-cost abstractions for system monitoring
const SystemMetrics = struct {
    cpu_usage: f64,
    memory_usage: f64,
    disk_io_read: u64,
    disk_io_write: u64,
    network_rx: u64,
    network_tx: u64,
    open_files: u64,
    active_connections: u64,
    timestamp: u64,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .cpu_usage = 0.0,
            .memory_usage = 0.0,
            .disk_io_read = 0,
            .disk_io_write = 0,
            .network_rx = 0,
            .network_tx = 0,
            .open_files = 0,
            .active_connections = 0,
            .timestamp = std.time.timestamp(),
        };
    }

    pub fn toJson(self: Self, allocator: Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\{{"cpu_usage":{d:.2},"memory_usage":{d:.2},"disk_io_read":{},"disk_io_write":{},"network_rx":{},"network_tx":{},"open_files":{},"active_connections":{},"timestamp":{}}}
        , .{ self.cpu_usage, self.memory_usage, self.disk_io_read, self.disk_io_write, self.network_rx, self.network_tx, self.open_files, self.active_connections, self.timestamp });
    }
};

// High-performance file operations with zero-copy where possible
const FileManager = struct {
    allocator: Allocator,
    open_files: HashMap([]const u8, std.fs.File, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    file_cache: HashMap([]const u8, []u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .open_files = HashMap([]const u8, std.fs.File, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .file_cache = HashMap([]const u8, []u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Close all open files
        var iterator = self.open_files.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.close();
        }
        self.open_files.deinit();

        // Free cached file contents
        var cache_iterator = self.file_cache.iterator();
        while (cache_iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.file_cache.deinit();
    }

    pub fn readFileOptimized(self: *Self, file_path: []const u8) ![]u8 {
        // Check cache first
        if (self.file_cache.get(file_path)) |cached_content| {
            log(.info, "Cache hit for file: {s}", .{file_path});
            return cached_content;
        }

        // Open file with optimized flags
        const file = std.fs.cwd().openFile(file_path, .{ .mode = .read_only }) catch |err| {
            log(.err, "Failed to open file {s}: {}", .{ file_path, err });
            return err;
        };
        defer file.close();

        // Get file size for optimal allocation
        const file_size = try file.getEndPos();
        if (file_size > config.max_file_size) {
            log(.err, "File too large: {s} ({} bytes)", .{ file_path, file_size });
            return error.FileTooLarge;
        }

        // Allocate buffer and read entire file
        const content = try self.allocator.alloc(u8, file_size);
        const bytes_read = try file.readAll(content);

        if (bytes_read != file_size) {
            self.allocator.free(content);
            return error.IncompleteRead;
        }

        // Cache the content
        const owned_path = try self.allocator.dupe(u8, file_path);
        try self.file_cache.put(owned_path, content);

        log(.info, "File read and cached: {s} ({} bytes)", .{ file_path, file_size });
        return content;
    }

    pub fn writeFileOptimized(self: *Self, file_path: []const u8, content: []const u8) !void {
        const file = std.fs.cwd().createFile(file_path, .{ .truncate = true }) catch |err| {
            log(.err, "Failed to create file {s}: {}", .{ file_path, err });
            return err;
        };
        defer file.close();

        try file.writeAll(content);

        // Update cache
        const owned_path = try self.allocator.dupe(u8, file_path);
        const cached_content = try self.allocator.dupe(u8, content);
        try self.file_cache.put(owned_path, cached_content);

        log(.info, "File written and cached: {s} ({} bytes)", .{ file_path, content.len });
    }

    pub fn memoryMapFile(self: *Self, file_path: []const u8) ![]align(std.mem.page_size) const u8 {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        if (file_size > config.max_file_size) {
            return error.FileTooLarge;
        }

        // Memory map the file for zero-copy access
        const mapped_memory = try std.os.mmap(
            null,
            file_size,
            std.os.PROT.READ,
            std.os.MAP.PRIVATE,
            file.handle,
            0,
        );

        log(.info, "Memory mapped file: {s} ({} bytes)", .{ file_path, file_size });
        return mapped_memory;
    }
};

// System monitoring with compile-time optimizations
const SystemMonitor = struct {
    allocator: Allocator,
    metrics_history: ArrayList(SystemMetrics),
    running: Atomic(bool),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .metrics_history = ArrayList(SystemMetrics).init(allocator),
            .running = Atomic(bool).init(false),
        };
    }

    pub fn deinit(self: *Self) void {
        self.metrics_history.deinit();
    }

    pub fn start(self: *Self) !void {
        self.running.store(true, .SeqCst);
        log(.info, "System monitoring started");

        while (self.running.load(.SeqCst)) {
            const metrics = try self.collectMetrics();
            try self.metrics_history.append(metrics);

            // Keep only last 100 metrics to prevent memory growth
            if (self.metrics_history.items.len > 100) {
                _ = self.metrics_history.orderedRemove(0);
            }

            std.time.sleep(MONITORING_INTERVAL_MS * std.time.ns_per_ms);
        }
    }

    pub fn stop(self: *Self) void {
        self.running.store(false, .SeqCst);
        log(.info, "System monitoring stopped");
    }

    fn collectMetrics(self: *Self) !SystemMetrics {
        _ = self;
        var metrics = SystemMetrics.init();

        // CPU usage calculation
        metrics.cpu_usage = try getCpuUsage();

        // Memory usage calculation
        metrics.memory_usage = try getMemoryUsage();

        // Disk I/O statistics
        const disk_stats = try getDiskStats();
        metrics.disk_io_read = disk_stats.read_bytes;
        metrics.disk_io_write = disk_stats.write_bytes;

        // Network statistics
        const network_stats = try getNetworkStats();
        metrics.network_rx = network_stats.rx_bytes;
        metrics.network_tx = network_stats.tx_bytes;

        // File descriptors
        metrics.open_files = try getOpenFileCount();

        metrics.timestamp = @intCast(std.time.timestamp());

        return metrics;
    }

    pub fn getLatestMetrics(self: *Self) ?SystemMetrics {
        if (self.metrics_history.items.len == 0) return null;
        return self.metrics_history.items[self.metrics_history.items.len - 1];
    }

    pub fn getMetricsHistory(self: *Self) []const SystemMetrics {
        return self.metrics_history.items;
    }
};

// Network utilities for high-performance I/O
const NetworkStats = struct {
    rx_bytes: u64,
    tx_bytes: u64,
    rx_packets: u64,
    tx_packets: u64,
};

const DiskStats = struct {
    read_bytes: u64,
    write_bytes: u64,
    read_ops: u64,
    write_ops: u64,
};

// Platform-specific system information gathering
fn getCpuUsage() !f64 {
    // This would read from /proc/stat on Linux
    // For now, return simulated value
    return 15.5 + (@as(f64, @floatFromInt(std.crypto.random.int(u8))) / 255.0) * 20.0;
}

fn getMemoryUsage() !f64 {
    // This would read from /proc/meminfo on Linux
    // For now, return simulated value
    return 45.2 + (@as(f64, @floatFromInt(std.crypto.random.int(u8))) / 255.0) * 30.0;
}

fn getDiskStats() !DiskStats {
    // This would read from /proc/diskstats on Linux
    return DiskStats{
        .read_bytes = 1024 * 1024 * 100, // 100MB
        .write_bytes = 1024 * 1024 * 50, // 50MB
        .read_ops = 1000,
        .write_ops = 500,
    };
}

fn getNetworkStats() !NetworkStats {
    // This would read from /proc/net/dev on Linux
    return NetworkStats{
        .rx_bytes = 1024 * 1024 * 200, // 200MB
        .tx_bytes = 1024 * 1024 * 150, // 150MB
        .rx_packets = 10000,
        .tx_packets = 8000,
    };
}

fn getOpenFileCount() !u64 {
    // This would check /proc/sys/fs/file-nr on Linux
    return 42;
}

// HTTP server implementation
const HttpServer = struct {
    allocator: Allocator,
    file_manager: FileManager,
    system_monitor: SystemMonitor,
    monitor_thread: ?Thread,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .file_manager = FileManager.init(allocator),
            .system_monitor = SystemMonitor.init(allocator),
            .monitor_thread = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.monitor_thread) |thread| {
            self.system_monitor.stop();
            thread.join();
        }
        self.system_monitor.deinit();
        self.file_manager.deinit();
    }

    pub fn start(self: *Self) !void {
        log(.info, "🔧 ZIG SYSTEM UTILS SERVICE INDÍTÁSA");
        log(.info, "Port: {d}", .{PORT});
        log(.info, "Max connections: {d}", .{MAX_CONNECTIONS});
        log(.info, "Buffer size: {d} bytes", .{BUFFER_SIZE});

        // Start system monitoring in background
        self.monitor_thread = try Thread.spawn(.{}, systemMonitorWorker, .{&self.system_monitor});

        // Start HTTP server
        var server = std.net.StreamServer.init(.{ .reuse_address = true });
        defer server.deinit();

        const address = std.net.Address.parseIp("0.0.0.0", PORT) catch unreachable;
        try server.listen(address);

        log(.info, "✅ Zig System Utils service listening on port {d}", .{PORT});
        log(.info, "Supported features: File I/O optimization, Memory mapping, System monitoring");

        while (true) {
            const connection = server.accept() catch |err| {
                log(.err, "Failed to accept connection: {}", .{err});
                continue;
            };

            // Handle each connection in a separate thread
            _ = Thread.spawn(.{}, handleConnection, .{ self, connection }) catch |err| {
                log(.err, "Failed to spawn connection handler: {}", .{err});
                connection.stream.close();
            };
        }
    }

    fn handleConnection(self: *Self, connection: std.net.StreamServer.Connection) void {
        defer connection.stream.close();

        var buffer: [BUFFER_SIZE]u8 = undefined;
        const bytes_read = connection.stream.read(buffer[0..]) catch |err| {
            log(.err, "Failed to read from connection: {}", .{err});
            return;
        };

        const request = buffer[0..bytes_read];
        self.processRequest(connection.stream, request) catch |err| {
            log(.err, "Failed to process request: {}", .{err});
        };
    }

    fn processRequest(self: *Self, stream: std.net.Stream, request: []const u8) !void {
        // Parse HTTP request
        const request_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return error.InvalidRequest;
        const request_line = request[0..request_line_end];

        var parts = std.mem.split(u8, request_line, " ");
        const method = parts.next() orelse return error.InvalidRequest;
        const path = parts.next() orelse return error.InvalidRequest;

        if (std.mem.eql(u8, method, "GET")) {
            try self.handleGet(stream, path);
        } else if (std.mem.eql(u8, method, "POST")) {
            try self.handlePost(stream, path, request);
        } else {
            try self.sendError(stream, 405, "Method Not Allowed");
        }
    }

    fn handleGet(self: *Self, stream: std.net.Stream, path: []const u8) !void {
        if (std.mem.eql(u8, path, "/health")) {
            const health_json = 
                \\{"status":"healthy","service":"zig-utils","timestamp":
            ++ std.fmt.allocPrint(self.allocator, "{d}", .{std.time.timestamp()}) catch "0" ++
                \\,"features":["file_io","memory_mapping","system_monitoring"]}
            ;
            defer self.allocator.free(health_json);
            try self.sendResponse(stream, 200, "application/json", health_json);
        } else if (std.mem.eql(u8, path, "/metrics")) {
            if (self.system_monitor.getLatestMetrics()) |metrics| {
                const metrics_json = try metrics.toJson(self.allocator);
                defer self.allocator.free(metrics_json);
                try self.sendResponse(stream, 200, "application/json", metrics_json);
            } else {
                try self.sendError(stream, 503, "Metrics not available");
            }
        } else if (std.mem.eql(u8, path, "/metrics/history")) {
            const history = self.system_monitor.getMetricsHistory();
            var json_buffer = ArrayList(u8).init(self.allocator);
            defer json_buffer.deinit();

            try json_buffer.appendSlice("[");
            for (history, 0..) |metrics, i| {
                if (i > 0) try json_buffer.appendSlice(",");
                const metrics_json = try metrics.toJson(self.allocator);
                defer self.allocator.free(metrics_json);
                try json_buffer.appendSlice(metrics_json);
            }
            try json_buffer.appendSlice("]");

            try self.sendResponse(stream, 200, "application/json", json_buffer.items);
        } else {
            try self.sendError(stream, 404, "Not Found");
        }
    }

    fn handlePost(self: *Self, stream: std.net.Stream, path: []const u8, request: []const u8) !void {
        // Find request body
        const body_start = std.mem.indexOf(u8, request, "\r\n\r\n");
        if (body_start == null) {
            try self.sendError(stream, 400, "Bad Request - No body");
            return;
        }

        const body = request[body_start.? + 4 ..];

        if (std.mem.eql(u8, path, "/file/read")) {
            try self.handleFileRead(stream, body);
        } else if (std.mem.eql(u8, path, "/file/write")) {
            try self.handleFileWrite(stream, body);
        } else if (std.mem.eql(u8, path, "/file/mmap")) {
            try self.handleFileMemoryMap(stream, body);
        } else {
            try self.sendError(stream, 404, "Not Found");
        }
    }

    fn handleFileRead(self: *Self, stream: std.net.Stream, body: []const u8) !void {
        // Parse JSON body (simplified)
        const file_path = extractJsonString(body, "file_path") orelse {
            try self.sendError(stream, 400, "Missing file_path");
            return;
        };

        const content = self.file_manager.readFileOptimized(file_path) catch |err| {
            const error_msg = switch (err) {
                error.FileNotFound => "File not found",
                error.FileTooLarge => "File too large",
                error.AccessDenied => "Access denied",
                else => "Read error",
            };
            try self.sendError(stream, 500, error_msg);
            return;
        };

        const response_json = try std.fmt.allocPrint(self.allocator,
            \\{{"status":"success","file_path":"{s}","size":{},"content_preview":"{s}"}}
        , .{ file_path, content.len, if (content.len > 100) content[0..100] else content });
        defer self.allocator.free(response_json);

        try self.sendResponse(stream, 200, "application/json", response_json);
    }

    fn handleFileWrite(self: *Self, stream: std.net.Stream, body: []const u8) !void {
        const file_path = extractJsonString(body, "file_path") orelse {
            try self.sendError(stream, 400, "Missing file_path");
            return;
        };

        const content = extractJsonString(body, "content") orelse {
            try self.sendError(stream, 400, "Missing content");
            return;
        };

        self.file_manager.writeFileOptimized(file_path, content) catch |err| {
            const error_msg = switch (err) {
                error.AccessDenied => "Access denied",
                error.NoSpaceLeft => "No space left",
                else => "Write error",
            };
            try self.sendError(stream, 500, error_msg);
            return;
        };

        const response_json = try std.fmt.allocPrint(self.allocator,
            \\{{"status":"success","file_path":"{s}","bytes_written":{}}}
        , .{ file_path, content.len });
        defer self.allocator.free(response_json);

        try self.sendResponse(stream, 200, "application/json", response_json);
    }

    fn handleFileMemoryMap(self: *Self, stream: std.net.Stream, body: []const u8) !void {
        const file_path = extractJsonString(body, "file_path") orelse {
            try self.sendError(stream, 400, "Missing file_path");
            return;
        };

        const mapped_memory = self.file_manager.memoryMapFile(file_path) catch |err| {
            const error_msg = switch (err) {
                error.FileNotFound => "File not found",
                error.FileTooLarge => "File too large",
                error.OutOfMemory => "Out of memory",
                else => "Memory mapping error",
            };
            try self.sendError(stream, 500, error_msg);
            return;
        };

        const response_json = try std.fmt.allocPrint(self.allocator,
            \\{{"status":"success","file_path":"{s}","mapped_size":{},"address":"0x{x}"}}
        , .{ file_path, mapped_memory.len, @intFromPtr(mapped_memory.ptr) });
        defer self.allocator.free(response_json);

        try self.sendResponse(stream, 200, "application/json", response_json);
    }

    fn sendResponse(self: *Self, stream: std.net.Stream, status_code: u16, content_type: []const u8, body: []const u8) !void {
        _ = self;
        const response = try std.fmt.allocPrint(self.allocator,
            "HTTP/1.1 {d} OK\r\nContent-Type: {s}\r\nContent-Length: {d}\r\n\r\n{s}",
            .{ status_code, content_type, body.len, body },
        );
        defer self.allocator.free(response);

        _ = try stream.writeAll(response);
    }

    fn sendError(self: *Self, stream: std.net.Stream, status_code: u16, message: []const u8) !void {
        const error_json = try std.fmt.allocPrint(self.allocator, 
            \\{{"error":"{s}","status_code":{}}}
        , .{ message, status_code });
        defer self.allocator.free(error_json);

        try self.sendResponse(stream, status_code, "application/json", error_json);
    }
};

// Utility functions
fn systemMonitorWorker(monitor: *SystemMonitor) void {
    monitor.start() catch |err| {
        log(.err, "System monitor failed: {}", .{err});
    };
}

fn extractJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    // Simplified JSON string extraction
    const key_pattern = "\"" ++ key ++ "\":\"";
    const start_pos = std.mem.indexOf(u8, json, key_pattern) orelse return null;
    const value_start = start_pos + key_pattern.len;
    const value_end = std.mem.indexOfPos(u8, json, value_start, "\"") orelse return null;
    return json[value_start..value_end];
}

fn log(level: LogLevel, comptime format: []const u8, args: anytype) void {
    if (@intFromEnum(level) >= @intFromEnum(config.log_level)) {
        const level_str = switch (level) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .err => "ERROR",
        };
        print("[{s}] " ++ format ++ "\n", .{level_str} ++ args);
    }
}

// Main entry point
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = HttpServer.init(allocator);
    defer server.deinit();

    try server.start();
}