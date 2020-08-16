(define (add) (lambda (x) (lambda (y) (delay (+ (force x) (force y))))))
(define (mul) (lambda (x) (lambda (y) (delay (* (force x) (force y))))))
(define (div) (lambda (x) (lambda (y) (delay (quotient (force x) (force y))))))
(define (lt) (lambda (x) (lambda (y) (delay (if (< (force x) (force y)) (t) (f))))))
(define (eq) (lambda (x) (lambda (y) (delay (if (= (force x) (force y)) (t) (f))))))
(define (neg) (lambda (x) (delay (- (force x)))))

(define (s) (lambda (x) (lambda (y) (lambda (z) (delay (force ((force ((force x) z)) ((force y) z))))))))
(define (c) (lambda (x) (lambda (y) (lambda (z) (delay (force ((force ((force x) z)) y)))))))
(define (b) (lambda (x) (lambda (y) (lambda (z) (delay (force ((force x) ((force y) z))))))))

(define (i) (lambda (x) x))

(define (f) (lambda (x) (lambda (y) y)))
(define (t) (lambda (x) (lambda (y) x)))

(define (mycons) (lambda (x) (lambda (y) (lambda (z) (delay (force ((force ((force z) x)) y)))))))
(define (mycar) (lambda (c) (delay (force ((force c) (t))))))
(define (mycdr) (lambda (c) (delay (force ((force c) (f))))))

(define (nil) (lambda (x) (t)))
(define (isnil) (lambda (x) ((force x) (lambda (x) (lambda (y) (f))))))
