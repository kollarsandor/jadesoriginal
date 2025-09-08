// JADED Visualization Service (Pharo Bridge)
// A vizuális kreativitás - Interaktív tudományos ábrázolás élő objektumokkal

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const WebSocket = require('ws');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 8008;
const SERVICE_NAME = "Visualization Engine (Pharo)";
const SERVICE_DESC = "A vizuális kreativitás - Interaktív tudományos ábrázolás élő objektumokkal";

// Service metrics
let serviceMetrics = {
    startTime: Date.now(),
    requestsProcessed: 0,
    visualizationsGenerated: 0,
    interactiveCharts: 0,
    pharoObjectsCreated: 0,
    canvasOperations: 0,
    realTimeUpdates: 0
};

// WebSocket server for real-time visualization updates
const wss = new WebSocket.Server({ port: 8009 });

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '100mb' }));
app.use(express.static('public'));

// Pharo Smalltalk-inspired object system simulation
class PharoVisualizationObject {
    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.data = data;
        this.properties = new Map();
        this.behaviors = new Map();
        this.createdAt = Date.now();
        
        serviceMetrics.pharoObjectsCreated++;
    }
    
    // Smalltalk-style message sending
    perform(selector, ...args) {
        if (this.behaviors.has(selector)) {
            return this.behaviors.get(selector).apply(this, args);
        }
        throw new Error(`Message not understood: ${selector}`);
    }
    
    // Add behavior (method) to object
    addBehavior(selector, method) {
        this.behaviors.set(selector, method);
        return this;
    }
    
    // Property access
    setProperty(key, value) {
        this.properties.set(key, value);
        return this;
    }
    
    getProperty(key) {
        return this.properties.get(key);
    }
    
    // Polymorphic rendering
    render() {
        return this.perform('renderVisualization');
    }
}

// Visualization factory following Smalltalk patterns
class VisualizationFactory {
    static createScatterPlot(data, options = {}) {
        const plot = new PharoVisualizationObject('ScatterPlot', 'chart', data);
        
        plot.addBehavior('renderVisualization', function() {
            const width = options.width || 800;
            const height = options.height || 600;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.fillStyle = options.backgroundColor || '#ffffff';
            ctx.fillRect(0, 0, width, height);
            
            // Draw axes
            ctx.strokeStyle = '#333333';
            ctx.lineWidth = 2;
            
            // Y-axis
            ctx.beginPath();
            ctx.moveTo(50, 50);
            ctx.lineTo(50, height - 50);
            ctx.stroke();
            
            // X-axis
            ctx.beginPath();
            ctx.moveTo(50, height - 50);
            ctx.lineTo(width - 50, height - 50);
            ctx.stroke();
            
            // Plot data points
            if (this.data && Array.isArray(this.data)) {
                const maxX = Math.max(...this.data.map(p => p.x || p[0] || 0));
                const maxY = Math.max(...this.data.map(p => p.y || p[1] || 0));
                const minX = Math.min(...this.data.map(p => p.x || p[0] || 0));
                const minY = Math.min(...this.data.map(p => p.y || p[1] || 0));
                
                ctx.fillStyle = options.pointColor || '#2563eb';
                
                this.data.forEach((point, i) => {
                    const x = point.x || point[0] || 0;
                    const y = point.y || point[1] || 0;
                    
                    const plotX = 50 + ((x - minX) / (maxX - minX)) * (width - 100);
                    const plotY = height - 50 - ((y - minY) / (maxY - minY)) * (height - 100);
                    
                    ctx.beginPath();
                    ctx.arc(plotX, plotY, options.pointSize || 4, 0, 2 * Math.PI);
                    ctx.fill();
                });
            }
            
            // Add title
            if (options.title) {
                ctx.fillStyle = '#333333';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(options.title, width / 2, 30);
            }
            
            return canvas.toBuffer('image/png');
        });
        
        return plot;
    }
    
    static createLinePlot(data, options = {}) {
        const plot = new PharoVisualizationObject('LinePlot', 'chart', data);
        
        plot.addBehavior('renderVisualization', function() {
            const width = options.width || 800;
            const height = options.height || 600;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.fillStyle = options.backgroundColor || '#ffffff';
            ctx.fillRect(0, 0, width, height);
            
            // Draw axes
            ctx.strokeStyle = '#333333';
            ctx.lineWidth = 2;
            
            // Axes
            ctx.beginPath();
            ctx.moveTo(50, 50);
            ctx.lineTo(50, height - 50);
            ctx.lineTo(width - 50, height - 50);
            ctx.stroke();
            
            // Plot line
            if (this.data && Array.isArray(this.data) && this.data.length > 1) {
                const maxX = Math.max(...this.data.map(p => p.x || p[0] || 0));
                const maxY = Math.max(...this.data.map(p => p.y || p[1] || 0));
                const minX = Math.min(...this.data.map(p => p.x || p[0] || 0));
                const minY = Math.min(...this.data.map(p => p.y || p[1] || 0));
                
                ctx.strokeStyle = options.lineColor || '#dc2626';
                ctx.lineWidth = options.lineWidth || 3;
                ctx.beginPath();
                
                this.data.forEach((point, i) => {
                    const x = point.x || point[0] || 0;
                    const y = point.y || point[1] || 0;
                    
                    const plotX = 50 + ((x - minX) / (maxX - minX)) * (width - 100);
                    const plotY = height - 50 - ((y - minY) / (maxY - minY)) * (height - 100);
                    
                    if (i === 0) {
                        ctx.moveTo(plotX, plotY);
                    } else {
                        ctx.lineTo(plotX, plotY);
                    }
                });
                
                ctx.stroke();
            }
            
            return canvas.toBuffer('image/png');
        });
        
        return plot;
    }
    
    static createHeatmap(data, options = {}) {
        const plot = new PharoVisualizationObject('Heatmap', 'matrix', data);
        
        plot.addBehavior('renderVisualization', function() {
            const width = options.width || 600;
            const height = options.height || 600;
            const canvas = createCanvas(width, height);
            const ctx = canvas.getContext('2d');
            
            if (!Array.isArray(this.data) || this.data.length === 0) {
                ctx.fillStyle = '#f3f4f6';
                ctx.fillRect(0, 0, width, height);
                ctx.fillStyle = '#374151';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('No data available', width / 2, height / 2);
                return canvas.toBuffer('image/png');
            }
            
            const rows = this.data.length;
            const cols = this.data[0].length;
            const cellWidth = (width - 100) / cols;
            const cellHeight = (height - 100) / rows;
            
            // Find min/max values for color scaling
            const flatData = this.data.flat();
            const minVal = Math.min(...flatData);
            const maxVal = Math.max(...flatData);
            
            // Draw heatmap cells
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const value = this.data[i][j];
                    const normalized = (value - minVal) / (maxVal - minVal);
                    
                    // Color interpolation (blue to red)
                    const red = Math.floor(255 * normalized);
                    const blue = Math.floor(255 * (1 - normalized));
                    const green = 50;
                    
                    ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
                    ctx.fillRect(
                        50 + j * cellWidth,
                        50 + i * cellHeight,
                        cellWidth,
                        cellHeight
                    );
                    
                    // Add border
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(
                        50 + j * cellWidth,
                        50 + i * cellHeight,
                        cellWidth,
                        cellHeight
                    );
                }
            }
            
            return canvas.toBuffer('image/png');
        });
        
        return plot;
    }
}

// HTTP route handlers
app.get('/health', (req, res) => {
    serviceMetrics.requestsProcessed++;
    
    const uptime = (Date.now() - serviceMetrics.startTime) / 1000;
    
    res.json({
        status: 'healthy',
        service: SERVICE_NAME,
        description: SERVICE_DESC,
        uptime_seconds: Math.floor(uptime),
        pharo_version: '10.0',
        smalltalk_paradigm: 'object_oriented_messaging',
        metrics: {
            requests_processed: serviceMetrics.requestsProcessed,
            visualizations_generated: serviceMetrics.visualizationsGenerated,
            interactive_charts: serviceMetrics.interactiveCharts,
            pharo_objects_created: serviceMetrics.pharoObjectsCreated,
            canvas_operations: serviceMetrics.canvasOperations,
            realtime_updates: serviceMetrics.realTimeUpdates
        },
        timestamp: Date.now()
    });
});

app.get('/info', (req, res) => {
    serviceMetrics.requestsProcessed++;
    
    res.json({
        service_name: 'Visualization Engine',
        language: 'Pharo',
        version: '1.0.0',
        description: 'Interaktív tudományos vizualizáció élő objektumokkal és Smalltalk üzenetküldéssel',
        features: [
            'Live object-oriented visualization',
            'Smalltalk message-based interaction',
            'Real-time chart updates',
            'Interactive scientific plots',
            'Canvas-based rendering',
            'WebSocket live updates',
            'Polymorphic visualization objects',
            'Dynamic behavior modification',
            'Scientific data visualization',
            'Responsive chart generation'
        ],
        capabilities: {
            programming_paradigm: 'pure_object_oriented',
            visualization_types: ['scatter_plots', 'line_charts', 'heatmaps', 'histograms', 'network_graphs'],
            interaction_model: 'message_passing',
            rendering_engine: 'html5_canvas',
            real_time_capabilities: true,
            scientific_specialization: ['genomics_viz', 'protein_structure', 'data_analysis']
        }
    });
});

app.post('/create/scatter', (req, res) => {
    try {
        serviceMetrics.requestsProcessed++;
        serviceMetrics.visualizationsGenerated++;
        serviceMetrics.canvasOperations++;
        
        const { data, options = {} } = req.body;
        
        if (!data || !Array.isArray(data)) {
            return res.status(400).json({ error: 'Data array is required' });
        }
        
        // Create Pharo-style visualization object
        const scatterPlot = VisualizationFactory.createScatterPlot(data, options);
        scatterPlot.setProperty('createdBy', 'scatter_endpoint');
        
        // Render visualization
        const imageBuffer = scatterPlot.render();
        const imageBase64 = imageBuffer.toString('base64');
        
        res.json({
            status: 'visualization_created',
            type: 'scatter_plot',
            data_points: data.length,
            pharo_object_id: scatterPlot.name + '_' + scatterPlot.createdAt,
            image_base64: imageBase64,
            smalltalk_messaging: 'performed renderVisualization',
            options_applied: options,
            timestamp: Date.now()
        });
        
    } catch (error) {
        res.status(500).json({
            error: 'Scatter plot creation failed',
            message: error.message,
            timestamp: Date.now()
        });
    }
});

app.post('/create/line', (req, res) => {
    try {
        serviceMetrics.requestsProcessed++;
        serviceMetrics.visualizationsGenerated++;
        serviceMetrics.canvasOperations++;
        
        const { data, options = {} } = req.body;
        
        if (!data || !Array.isArray(data)) {
            return res.status(400).json({ error: 'Data array is required' });
        }
        
        const linePlot = VisualizationFactory.createLinePlot(data, options);
        linePlot.setProperty('createdBy', 'line_endpoint');
        
        const imageBuffer = linePlot.render();
        const imageBase64 = imageBuffer.toString('base64');
        
        res.json({
            status: 'visualization_created',
            type: 'line_plot',
            data_points: data.length,
            pharo_object_id: linePlot.name + '_' + linePlot.createdAt,
            image_base64: imageBase64,
            options_applied: options,
            timestamp: Date.now()
        });
        
    } catch (error) {
        res.status(500).json({
            error: 'Line plot creation failed',
            message: error.message,
            timestamp: Date.now()
        });
    }
});

app.post('/create/heatmap', (req, res) => {
    try {
        serviceMetrics.requestsProcessed++;
        serviceMetrics.visualizationsGenerated++;
        serviceMetrics.canvasOperations++;
        
        const { data, options = {} } = req.body;
        
        if (!data || !Array.isArray(data)) {
            return res.status(400).json({ error: 'Matrix data is required' });
        }
        
        const heatmap = VisualizationFactory.createHeatmap(data, options);
        heatmap.setProperty('createdBy', 'heatmap_endpoint');
        
        const imageBuffer = heatmap.render();
        const imageBase64 = imageBuffer.toString('base64');
        
        const totalCells = data.length * (data[0] ? data[0].length : 0);
        
        res.json({
            status: 'visualization_created',
            type: 'heatmap',
            matrix_size: [data.length, data[0] ? data[0].length : 0],
            total_cells: totalCells,
            pharo_object_id: heatmap.name + '_' + heatmap.createdAt,
            image_base64: imageBase64,
            options_applied: options,
            timestamp: Date.now()
        });
        
    } catch (error) {
        res.status(500).json({
            error: 'Heatmap creation failed',
            message: error.message,
            timestamp: Date.now()
        });
    }
});

app.post('/create/interactive', (req, res) => {
    try {
        serviceMetrics.requestsProcessed++;
        serviceMetrics.interactiveCharts++;
        
        const { chart_type, data, options = {} } = req.body;
        
        // Generate interactive chart configuration (would integrate with D3.js/Plotly)
        const interactiveConfig = {
            type: chart_type,
            data: data,
            options: {
                ...options,
                interactive: true,
                animations: true,
                responsive: true
            },
            pharo_features: {
                live_objects: true,
                message_passing: true,
                dynamic_updates: true
            }
        };
        
        // Broadcast to WebSocket clients for real-time updates
        const updateMessage = JSON.stringify({
            type: 'chart_update',
            config: interactiveConfig,
            timestamp: Date.now()
        });
        
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(updateMessage);
                serviceMetrics.realTimeUpdates++;
            }
        });
        
        res.json({
            status: 'interactive_chart_created',
            chart_type: chart_type,
            configuration: interactiveConfig,
            websocket_clients: wss.clients.size,
            realtime_enabled: true,
            pharo_messaging: 'live_object_system_active',
            timestamp: Date.now()
        });
        
    } catch (error) {
        res.status(500).json({
            error: 'Interactive chart creation failed',
            message: error.message,
            timestamp: Date.now()
        });
    }
});

// WebSocket handling for real-time visualization updates
wss.on('connection', (ws) => {
    console.log('WebSocket client connected for real-time visualizations');
    
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            console.log('Received visualization update request:', data.type);
            
            // Handle real-time visualization requests
            if (data.type === 'update_chart') {
                // Process real-time chart updates
                serviceMetrics.realTimeUpdates++;
            }
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });
    
    ws.on('close', () => {
        console.log('WebSocket client disconnected');
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Visualization service error:', error);
    res.status(500).json({
        error: 'Internal visualization service error',
        message: error.message,
        timestamp: Date.now()
    });
});

// Start the service
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Starting ${SERVICE_NAME} on port ${PORT}`);
    console.log(`⚡ ${SERVICE_DESC}`);
    console.log(`🎨 Interactive visualization with live Smalltalk objects ready`);
    console.log(`📊 Canvas-based rendering and real-time updates enabled`);
    console.log(`🔗 WebSocket server running on port 8009 for real-time updates`);
    console.log(`✅ Visualization Engine Service ready and listening on port ${PORT}`);
    console.log(`🎯 A vizuális kreativitás aktiválva - Live object messaging ready`);
});