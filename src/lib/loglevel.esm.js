/*
 * Minimal browser ESM shim for the `loglevel` npm package.
 *
 * The npm package only ships a UMD/CommonJS build, which a browser cannot load
 * via a bare `import log from "loglevel"`. The web visualizer pages only use a
 * small subset of the API (log.trace/debug/info/warn/error plus level control),
 * so this shim backs those calls with `console` and defaults to WARN — matching
 * loglevel's own default of suppressing info/debug output.
 *
 * Node usage (the simulation/runner code) keeps using the real `loglevel`
 * package from node_modules; this file is only mapped in for the browser via
 * the import maps in web/*.html.
 */
const LEVELS = { TRACE: 0, DEBUG: 1, INFO: 2, WARN: 3, ERROR: 4, SILENT: 5 };

let currentLevel = LEVELS.WARN;

function emit(consoleMethod, messageLevel) {
	return (...args) => {
		if (messageLevel >= currentLevel) {
			(console[consoleMethod] || console.log).apply(console, args);
		}
	};
}

const log = {
	levels: LEVELS,
	trace: emit('trace', LEVELS.TRACE),
	debug: emit('debug', LEVELS.DEBUG),
	info: emit('info', LEVELS.INFO),
	warn: emit('warn', LEVELS.WARN),
	error: emit('error', LEVELS.ERROR),
	getLevel() {
		return currentLevel;
	},
	setLevel(level) {
		if (typeof level === 'string') level = LEVELS[level.toUpperCase()];
		if (typeof level === 'number') currentLevel = level;
		return currentLevel;
	},
	setDefaultLevel(level) {
		return this.setLevel(level);
	},
	getLogger() {
		return log;
	},
};

export default log;
