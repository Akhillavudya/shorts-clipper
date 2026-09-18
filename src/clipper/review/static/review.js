/* Review page behaviour.
 *
 * Everything here is an enhancement: the forms post and the noscript <video>
 * plays without it. What it adds is not reloading the page between verdicts,
 * which is the difference between reviewing three clips and thirty.
 */
(function () {
  "use strict";

  var cards = Array.prototype.slice.call(document.querySelectorAll(".card[data-clip-id]"));
  if (!cards.length) return;

  var current = 0;

  /* -- verdicts ----------------------------------------------------- */

  var VERDICT_TO_APPROVED = { approve: true, reject: false, reset: null };

  function verdictOf(card) {
    return card.dataset.verdict || "pending";
  }

  function approvedOf(card) {
    var v = verdictOf(card);
    if (v === "approved") return true;
    if (v === "rejected") return false;
    return null;
  }

  function paint(card, data) {
    card.dataset.verdict = data.verdict;
    card.classList.remove("card--pending", "card--approved", "card--rejected");
    card.classList.add("card--" + data.verdict);

    var label = card.querySelector("[data-verdict-label]");
    if (label) label.textContent = data.verdict;

    card.querySelectorAll("[data-verdict]").forEach(function (button) {
      var pressed =
        (button.dataset.verdict === "approve" && data.approved === true) ||
        (button.dataset.verdict === "reject" && data.approved === false);
      button.setAttribute("aria-pressed", pressed ? "true" : "false");
    });

    var tally = document.getElementById("tally");
    if (tally && data.counts) {
      var words = { approved: "kept", rejected: "cut", pending: "to do" };
      Object.keys(words).forEach(function (key) {
        var node = tally.querySelector('[data-count="' + key + '"]');
        if (node) node.textContent = data.counts[key] + " " + words[key];
      });
    }
  }

  function announce(card, message) {
    var slot = card.querySelector("[data-saved]");
    if (!slot) return;
    slot.textContent = message;
    if (message) {
      window.setTimeout(function () {
        if (slot.textContent === message) slot.textContent = "";
      }, 2500);
    }
  }

  function send(card, approved) {
    var form = card.querySelector("[data-review-form]");
    var note = card.querySelector("[data-note]");
    announce(card, "saving...");

    return fetch(form.action, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ approved: approved, note: note ? note.value : null })
    })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        paint(card, data);
        announce(card, "saved");
      })
      .catch(function (error) {
        /* A failed write must not look like a saved one: say so and leave the
         * page alone, so a reload shows what is actually on disk. */
        announce(card, "not saved - " + error.message);
      });
  }

  cards.forEach(function (card, index) {
    var form = card.querySelector("[data-review-form]");
    if (!form) return;

    form.addEventListener("submit", function (event) {
      var submitter = event.submitter;
      var verdict = submitter && submitter.dataset ? submitter.dataset.verdict : null;
      if (!verdict) return; /* no submitter: let the browser post it */
      event.preventDefault();
      current = index;
      send(card, VERDICT_TO_APPROVED[verdict]);
    });

    var note = card.querySelector("[data-note]");
    if (note) {
      var saved = note.value;
      note.addEventListener("blur", function () {
        if (note.value === saved) return;
        saved = note.value;
        send(card, approvedOf(card));
      });
    }

    card.addEventListener("focusin", function () {
      current = index;
    });
  });

  /* -- previews ------------------------------------------------------ */

  /* The source fallback is one long file, so the element is created on demand
   * (a preloading <video> per card would range-request the whole episode three
   * times over) and it stops itself at the cut point. */
  function mount(frame) {
    var video = document.createElement("video");
    var start = parseFloat(frame.dataset.start) || 0;
    var end = parseFloat(frame.dataset.end) || 0;

    video.controls = true;
    video.autoplay = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = frame.dataset.src + (start ? "#t=" + start : "");

    video.addEventListener("loadedmetadata", function () {
      if (start && Math.abs(video.currentTime - start) > 0.25) video.currentTime = start;
    });

    if (end > start) {
      video.addEventListener("timeupdate", function () {
        if (video.currentTime >= end) {
          video.pause();
          video.currentTime = start;
        }
      });
    }

    video.addEventListener("play", function () {
      document.querySelectorAll("video").forEach(function (other) {
        if (other !== video) other.pause();
      });
    });

    frame.innerHTML = "";
    frame.appendChild(video);
    return video;
  }

  function play(card) {
    var frame = card.querySelector(".preview__frame");
    if (!frame) return;
    var video = frame.querySelector("video");
    if (!video) {
      mount(frame);
      return;
    }
    if (video.paused) video.play();
    else video.pause();
  }

  document.querySelectorAll("[data-play]").forEach(function (button) {
    button.addEventListener("click", function () {
      mount(button.closest(".preview__frame"));
    });
  });

  /* -- keyboard ------------------------------------------------------ */

  function focusCard(index) {
    current = Math.max(0, Math.min(cards.length - 1, index));
    cards[current].focus();
    cards[current].scrollIntoView({ block: "nearest" });
  }

  document.addEventListener("keydown", function (event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    var target = event.target;
    var typing =
      target &&
      (target.tagName === "TEXTAREA" ||
        target.tagName === "INPUT" ||
        target.isContentEditable);
    if (typing) return;

    var card = cards[current];

    switch (event.key) {
      case "j":
        event.preventDefault();
        focusCard(current + 1);
        break;
      case "k":
        event.preventDefault();
        focusCard(current - 1);
        break;
      case "a":
        event.preventDefault();
        send(card, true);
        break;
      case "r":
        event.preventDefault();
        send(card, false);
        break;
      case "u":
        event.preventDefault();
        send(card, null);
        break;
      case " ":
        /* Space on a button is that button's own activation, not ours. */
        if (target && target.tagName === "BUTTON") return;
        event.preventDefault();
        play(card);
        break;
      default:
        break;
    }
  });
})();
