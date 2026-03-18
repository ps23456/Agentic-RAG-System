import { useState, useEffect } from "react";

type Theme = "light" | "dark";

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = localStorage.getItem("isr-theme") as Theme | null;
    return saved ?? "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("isr-theme", theme);
  }, [theme]);

  const toggle = () => setThemeState((t) => (t === "light" ? "dark" : "light"));
  const setTheme = (t: Theme) => setThemeState(t);

  return { theme, toggle, setTheme };
}
