module.exports = {
    apps: [{
        name: "folding-api",
        script: "scripts/run_folding_api.sh", // Use the wrapper script
        autorestart: true,
        watch: false
    }]
};