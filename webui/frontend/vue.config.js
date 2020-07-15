module.exports = {
	outputDir: 'dist',
	assetsDir: './static',
	configureWebpack: {
		devServer: {
			proxy: "http://knowledge-glue-webui-jeffstudentsproject.ida.dcs.gla.ac.uk/"
		}
	}
}
