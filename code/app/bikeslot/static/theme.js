/* Light or dark: a stored choice is set as data-theme on <html>, without one the CSS follows the device.
   Loaded in <head> without defer, so the stored choice applies before the first paint. */
(() => {
  const KEY = "bikeslot-theme";
  const root = document.documentElement;
  const device = window.matchMedia("(prefers-color-scheme: dark)");

  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark") root.dataset.theme = stored;
  } catch {
    // storage blocked: follow the device
  }

  const current = () => root.dataset.theme || (device.matches ? "dark" : "light");
  const label = () => {
    const button = document.getElementById("theme-toggle");
    if (button) button.setAttribute("aria-label", current() === "dark" ? "Switch to light mode" : "Switch to dark mode");
  };
  device.addEventListener("change", label); // the CSS follows the device on its own, only the label changes

  document.addEventListener("DOMContentLoaded", () => {
    label();
    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      const theme = current() === "dark" ? "light" : "dark";
      root.dataset.theme = theme;
      label();
      try {
        localStorage.setItem(KEY, theme);
      } catch {
        // storage blocked: the choice lasts until the page is left
      }
    });
  });
})();
