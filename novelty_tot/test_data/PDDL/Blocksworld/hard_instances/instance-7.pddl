

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b f)
(ontable c)
(on d e)
(on e b)
(on f c)
(clear a)
(clear d)
)
(:goal
(and
(on a b)
(on c e)
(on e f)
(on f d))
)
)


