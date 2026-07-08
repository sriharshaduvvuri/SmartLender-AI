/**
 * Smart Lender - Core Frontend Scripts
 * Implements dark/light theme switching, local storage configuration,
 * interactive loaders, and validation helpers.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------------
    // Theme Switcher Logic
    // -----------------------------------------------------------------
    const themeToggleBtn = document.getElementById("theme-toggle");
    const currentTheme = localStorage.getItem("theme") || "light";

    // Set initial theme on boot
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeToggleIcon(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const activeTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = activeTheme === "dark" ? "light" : "dark";

            document.documentElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            updateThemeToggleIcon(newTheme);
            logger("Theme switched to: " + newTheme);
        });
    }

    function updateThemeToggleIcon(theme) {
        const icon = document.getElementById("theme-icon");
        if (icon) {
            if (theme === "dark") {
                icon.className = "bi bi-sun-fill text-warning"; // Sun for switching to light
            } else {
                icon.className = "bi bi-moon-stars-fill text-indigo"; // Moon for switching to dark
            }
        }
    }

    // -----------------------------------------------------------------
    // Form Submission Loader Animation
    // -----------------------------------------------------------------
    const predictionForm = document.getElementById("prediction-form");
    const loadingOverlay = document.getElementById("loading-overlay");

    if (predictionForm && loadingOverlay) {
        predictionForm.addEventListener("submit", (e) => {
            // Run basic validations first
            if (!validateClientInputs()) {
                e.preventDefault();
                return;
            }
            // Show loading animation on success
            loadingOverlay.style.display = "flex";
        });
    }

    // -----------------------------------------------------------------
    // Client-side Input Validations
    // -----------------------------------------------------------------
    function validateClientInputs() {
        const errors = [];
        
        // Income validation
        const appIncomeInput = document.getElementById("applicant_income");
        if (appIncomeInput) {
            const val = parseFloat(appIncomeInput.value);
            if (isNaN(val) || val < 0) {
                errors.push("Applicant Income cannot be negative.");
                highlightField(appIncomeInput, true);
            } else {
                highlightField(appIncomeInput, false);
            }
        }

        const coappIncomeInput = document.getElementById("coapplicant_income");
        if (coappIncomeInput) {
            const val = parseFloat(coappIncomeInput.value);
            if (isNaN(val) || val < 0) {
                errors.push("Co-Applicant Income cannot be negative.");
                highlightField(coappIncomeInput, true);
            } else {
                highlightField(coappIncomeInput, false);
            }
        }

        // Loan amount validation
        const loanAmtInput = document.getElementById("loan_amount");
        if (loanAmtInput) {
            const val = parseFloat(loanAmtInput.value);
            if (isNaN(val) || val <= 0) {
                errors.push("Loan Amount must be greater than zero.");
                highlightField(loanAmtInput, true);
            } else {
                highlightField(loanAmtInput, false);
            }
        }

        // Loan term validation
        const loanTermInput = document.getElementById("loan_amount_term");
        if (loanTermInput) {
            const val = parseFloat(loanTermInput.value);
            if (isNaN(val) || val <= 0) {
                errors.push("Loan Term must be greater than zero.");
                highlightField(loanTermInput, true);
            } else {
                highlightField(loanTermInput, false);
            }
        }

        // Print errors if found
        if (errors.length > 0) {
            const errorContainer = document.getElementById("js-validation-errors");
            const errorsContent = document.getElementById("js-errors-content");
            if (errorContainer && errorsContent) {
                errorsContent.innerHTML = "<strong>Validation Errors:</strong><ul class='mb-0 mt-1'>" + errors.map(err => `<li>${err}</li>`).join("") + "</ul>";
                errorContainer.style.display = "block";
                errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                alert("Validation Error:\n- " + errors.join("\n- "));
            }
            return false;
        }

        return true;
    }

    function highlightField(inputElement, hasError) {
        if (hasError) {
            inputElement.classList.add("is-invalid");
        } else {
            inputElement.classList.remove("is-invalid");
        }
    }

    function logger(msg) {
        console.log("[Smart Lender LOG] " + msg);
    }
});
