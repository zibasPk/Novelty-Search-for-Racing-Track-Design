import log from "loglevel";
import fs from "fs";
import path from "path";

export function initLogger({
  filePath,
  level = "info",
  withTimestamp = true,
  fileOnly = false
}) {
  // Ensure directory exists
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });

  const logFile = fs.createWriteStream(filePath, { flags: "a" });
  const originalFactory = log.methodFactory;

  log.methodFactory = function (methodName, logLevel, loggerName) {
    const rawMethod = originalFactory(methodName, logLevel, loggerName);

    return function (...args) {
      const ts = withTimestamp
        ? `[${new Date().toLocaleTimeString()}] `
        : "";

      const formattedArgs = args
        .map(a => (typeof a === "object" ? JSON.stringify(a) : String(a)))
        .join(" ");

      const line = `${ts}[${methodName.toUpperCase()}] ${formattedArgs}`;

      logFile.write(line + "\n");

      if (!fileOnly) rawMethod(line);
    };
  };

  log.setLevel(level);
  return log;
}